import os.path

import numpy as np
from box import Box

from data_loader import TwitterDataset
from llm import LLMClassifier, preprocessor, THRESHOLD


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


def precompute_ensemble_eval(config: Box) -> np.ndarray:
    """
    Compute the evaluation of the models on the validation set.
    Save model outputs and eval labels to files for further processing.
    :param config: Config object.
    """
    config.data.dedup_strategy = None # We don't care about train data

    outputs = []
    for checkpoint, base_model in zip(config.ensemble.models, config.ensemble.base_models):
        print(f"[+] Evaluating model {checkpoint}")
        checkpoint_path = os.path.join(config.ensemble.path, checkpoint)
        outputs.append(test_model(checkpoint_path, base_model, "eval", config))

    eval_outputs = np.column_stack(outputs)
    np.save("eval_outputs", eval_outputs)

    twitter = TwitterDataset(config)
    eval_labels = np.array(twitter.dataset["eval"]["label"])
    np.save("eval_labels", eval_labels)


if __name__ == "__main__":
    from utils import load_config, set_seed

    cfg = load_config()
    set_seed(cfg.general.seed)

    precompute_ensemble_eval(cfg)

    # cfg.data.max_samples = 1 # avoid loading training data

    # outputs = []
    # for checkpoint, base_model in zip(cfg.ensemble.models, cfg.ensemble.base_models):
    #     print(f"[+] Testing model {checkpoint}")
    #     checkpoint_path = os.path.join(cfg.data.checkpoint_path, checkpoint)
    #     outputs.append(test_model(checkpoint_path, base_model, "test", cfg))

    # outputs = np.array(outputs)
    # if cfg.ensemble.voting_strategy == "avg":
    #     result = np.mean(outputs, axis=0)
    #     result = np.where(result >= THRESHOLD, 1, -1)
    # elif cfg.ensemble.voting_strategy == "vote":
    #     result = np.where(outputs >= THRESHOLD, 1, -1)
    #     result = np.where(np.sum(result, axis=0) >= 0, 1, -1)

    # save_outputs(result, cfg.general.run_id)
