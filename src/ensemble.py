from typing import Tuple
import os.path

import numpy as np
from box import Box

from utils import THRESHOLD
from data_loader import TwitterDataset
from llm import LLMClassifier, preprocessor


def test_model(checkpoint_path: str, config: Box) -> Tuple[np.ndarray]:
    """
    Test the model on the 'eval' and 'test' splits.
    :param checkpoint_path: Path to the model checkpoint.
    :param config: Config object.
    :return: tuple of np.ndarray of shape (n_samples,) with test dataset labels.
    """
    twitter = TwitterDataset(config, preprocessor[config.llm.model])
    twitter.dataset['train'] = twitter.dataset['train'].select([0]) # avoid tokenizing train data

    llm = LLMClassifier(config)
    llm.load_checkpoint(checkpoint_path, config.llm.use_lora)

    tokenized_dataset = twitter.tokenize_to_hf(llm.tokenizer)
    eval_outputs = llm.test(tokenized_dataset['eval'], hard_labels=False)
    test_outputs = llm.test(tokenized_dataset['test'], hard_labels=False)

    eval_predictions = np.where(eval_outputs >= THRESHOLD, 1, 0)
    eval_labels = np.array(twitter.dataset["eval"]["label"])
    accuracy = np.mean(eval_predictions == eval_labels)
    print(f"[+] Evaluation accuracy: {accuracy:.3%}")

    del twitter
    del llm
    return eval_outputs, test_outputs


def precompute_ensemble_outputs(config: Box):
    """
    Compute and save to file the outputs for each ensemble model on the eval and test sets.
    :param config: Config object.
    """
    config.data.dedup_strategy = None # We don't care about train data

    eval_outputs, test_outputs = [], []
    for model in config.ensemble.models:
        checkpoint_path = os.path.join(config.ensemble.path, model.name)
        config.llm.model = model.base_model
        config.llm.max_len = model.max_len
        config.llm.use_lora = model.lora_r is not None
        config.llm.lora_r = model.lora_r
        config.llm.lora_alpha = model.lora_r

        print(f"\n[+] Running inference for model {model.name}.")
        eval_out, test_out = test_model(checkpoint_path, config)
        eval_outputs.append(eval_out)
        test_outputs.append(test_out)

    eval_outputs = np.column_stack(eval_outputs)
    test_outputs = np.column_stack(test_outputs)
    np.save(os.path.join(config.ensemble.path, "eval_outputs"), eval_outputs)
    np.save(os.path.join(config.ensemble.path, "test_outputs"), test_outputs)

    twitter = TwitterDataset(config)
    eval_labels = np.array(twitter.dataset["eval"]["label"])
    np.save(os.path.join(config.ensemble.path, "eval_labels"), eval_labels)


if __name__ == "__main__":
    from utils import load_config, set_seed

    cfg = load_config()
    set_seed(cfg.general.seed)

    precompute_ensemble_outputs(cfg)
