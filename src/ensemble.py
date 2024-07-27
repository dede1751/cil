import os.path

import numpy as np
from box import Box

from data_loader import TwitterDataset
from llm import LLMClassifier, preprocessor


def test_model(
    checkpoint_path: str,
    base_model: str,
    split: str,
    config: Box
) -> np.ndarray:
    """
    Test the model on the given dataset.
    :param checkpoint_path: Path to the model checkpoint.
    :param base_model: Base model name.
    :param split: Split to test the model on.
    :param config: Config object.
    :return: np.ndarray of shape (n_samples,) with test dataset labels.
    """
    config.llm.model = base_model

    twitter = TwitterDataset(config, preprocessor[config.llm.model])
    twitter.dataset['train'] = twitter.dataset['train'].select([0]) # avoid tokenizing train data

    llm = LLMClassifier(config)
    llm.load_checkpoint(checkpoint_path)

    tokenized_dataset = twitter.tokenize_to_hf(llm.tokenizer)
    outputs = llm.test(tokenized_dataset[split], hard_labels=False)

    del twitter
    del llm
    return outputs


def test_all_models(split: str, config: Box) -> np.ndarray:
    """
    Compute the evaluation of all the ensemble models on the specified split.
    :param config: Config object.
    :param split: Split to test the model on.
    :return: np.ndarray of shape (n_samples, n_models) with the model outputs.
    """
    config.data.dedup_strategy = None # We don't care about train data

    outputs = []
    for checkpoint, base_model in zip(config.ensemble.models, config.ensemble.base_models):
        print(f"\n[+] Evaluating model {checkpoint} on '{split}' set.")
        checkpoint_path = os.path.join(config.ensemble.path, checkpoint)
        outputs.append(test_model(checkpoint_path, base_model, split, config))

    return np.column_stack(outputs)


def precompute_ensemble_outputs(config: Box):
    """
    Compute and save to file the outputs for each ensemble model on the eval and test sets.
    :param config: Config object.
    """
    twitter = TwitterDataset(config)
    eval_labels = np.array(twitter.dataset["eval"]["label"])
    np.save(os.path.join(config.ensemble.path, "eval_labels"), eval_labels)

    eval_outputs = test_all_models("eval", config)
    test_outputs = test_all_models("test", config)
    np.save(os.path.join(config.ensemble.path, "eval_outputs"), eval_outputs)
    np.save(os.path.join(config.ensemble.path, "test_outputs"), test_outputs)


if __name__ == "__main__":
    from utils import load_config, set_seed

    cfg = load_config()
    set_seed(cfg.general.seed)

    precompute_ensemble_outputs(cfg)
