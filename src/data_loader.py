import os
from typing import Callable

from datasets import DatasetDict, Dataset
import pandas as pd
from box import Box
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


class TwitterDataset():
    """
    Custom dataset class for Twitter data. Provides datasets for the different ML frameworks.
    """
    def __init__(self, config: Box, preprocessor: Callable[[str], str] = None):
        """
        Load the dataset from the default data files. Performs train-test split.
        """
        self.cfg = config
        self.preprocessor = preprocessor
        self.dataset = self._load_dataset()
        for split in self.dataset:
            print(f"Number of rows in '{split}' dataset: {self.dataset[split].num_rows}")

    def _read_data(self, file: str, max_samples: int = None) -> pd.DataFrame:
        file_path = os.path.join(self.cfg.data.path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        tweets = [line.strip() for line in lines[:max_samples]]
        if self.preprocessor is not None:
            tweets = [self.preprocessor(tweet) for tweet in tweets]

        return pd.DataFrame(tweets, columns=['text'])

    def _load_dataset(self):
        if self.cfg.data.use_full_dataset:
            train_neg = self._read_data("train_neg_full.txt", self.cfg.data.max_samples)
            train_pos = self._read_data("train_pos_full.txt", self.cfg.data.max_samples)
        else:
            train_neg = self._read_data("train_neg.txt", self.cfg.data.max_samples)
            train_pos = self._read_data("train_pos.txt", self.cfg.data.max_samples)

        train_neg['label'] = 0.0
        train_pos['label'] = 1.0
        train_df = pd.concat([train_neg, train_pos], ignore_index=True)

        test_df = self._read_data("test_data.txt")
        test_df['text'] = test_df['text'].apply(lambda x: ",".join(x.split(",", 1)[1:]))

        train_df, eval_df = train_test_split(
            train_df, test_size=self.cfg.data.eval_split, random_state=self.cfg.general.seed)

        # hard: deduplicates and picks the most common label between 0 and 1
        # soft: deduplicates and averages the labels
        # smooth: deduplicates and smooths the labels with epsilon
        # null: no deduplication, labels are 0 and 1
        old_rows = train_df.shape[0]
        if self.cfg.data.dedup_strategy == "hard":
            train_df = train_df.groupby('text', as_index=False).mean()
            train_df['label'] = train_df['label'].apply(lambda x: 0 if x < 0.5 else 1.0)
        elif self.cfg.data.dedup_strategy == "soft":
            train_df = train_df.groupby('text', as_index=False).mean()
        elif self.cfg.data.dedup_strategy == "smooth":
            epsilon = self.cfg.data.smoothing_epsilon
            train_df = train_df.groupby('text', as_index=False).mean()
            train_df['label'] = train_df['label'].apply(lambda x: x * (1 - epsilon) + 0.5 * epsilon)

        print(f"[+] Removed {old_rows - train_df.shape[0]} duplicates.")
        train_df = train_df.reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)

        return DatasetDict(
            {'train': Dataset.from_pandas(train_df),
            'eval': Dataset.from_pandas(eval_df),
            'test': Dataset.from_pandas(test_df)})

    def tokenize_to_hf(
        self,
        tokenizer: AutoTokenizer,
    ) -> DatasetDict:
        """
        Tokenize the dataset using a HuggingFace AutoTokenizer.
        :param tokenizer: A HuggingFace AutoTokenizer object.
        :return: Tokenized split datasets ready for the HF Trainer
        """
        def tokenize_fn(examples):
            return tokenizer(
                examples['text'],
                max_length=self.cfg.llm.max_len,
                padding='max_length',
                truncation=True)

        print(f"[+] Tokenizing the dataset using {tokenizer.__class__.__name__}.")
        tokenized = self.dataset.map(tokenize_fn, batched=True)
        tokenized["train"].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        tokenized["eval"].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        tokenized["test"].set_format(type='torch', columns=['input_ids', 'attention_mask'])

        return tokenized


if __name__ == "__main__":
    from utils import load_config, set_seed

    cfg = load_config()
    set_seed(cfg.general.seed)
    twitter = TwitterDataset(cfg)
    tokenized_dataset = twitter.tokenize_to_hf(AutoTokenizer.from_pretrained(cfg.llm.model))
