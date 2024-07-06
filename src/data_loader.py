import os

from datasets import DatasetDict, Dataset
import pandas as pd
from box import Box
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


class TwitterDataset():
    """
    Custom dataset class for Twitter data. Provides datasets for the different ML frameworks.
    """
    def __init__(self, config: Box):
        """
        Load the dataset from the default data files. Performs train-test split.
        """
        self.cfg = config
        self.dataset = self._load_dataset()
        for split in self.dataset:
            print(f"Number of rows in '{split}' dataset: {self.dataset[split].num_rows}")

    def _read_data(self, file: str, max_samples: str = None) -> pd.DataFrame:
        file_path = os.path.join(self.cfg.data.path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if max_samples is not None:
            lines = lines[:max_samples]

        tweets = [line.strip() for line in lines]
        return pd.DataFrame(tweets, columns=['text'])

    def _load_dataset(self):
        max_samples = self.cfg.data.max_samples
        if self.cfg.data.use_full_dataset:
            train_neg = self._read_data("train_neg_full.txt", max_samples=max_samples)
            train_pos = self._read_data("train_pos_full.txt", max_samples=max_samples)
        else:
            train_neg = self._read_data("train_neg.txt", max_samples=max_samples)
            train_pos = self._read_data("train_pos.txt", max_samples=max_samples)

        train_neg['label'] = -1.0
        train_pos['label'] = 1.0
        train_df = pd.concat([train_neg, train_pos], ignore_index=True)

        test_df = self._read_data("test_data.txt")
        test_df['text'] = test_df['text'].apply(lambda x: ",".join(x.split(",", 1)[1:]))

        train_df, eval_df = train_test_split(
            train_df, test_size=0.2, random_state=self.cfg.general.seed)

        train_df = train_df.reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)

        return DatasetDict(
            {'train': Dataset.from_pandas(train_df),
            'eval': Dataset.from_pandas(eval_df),
            'test': Dataset.from_pandas(test_df)})

    def tokenize_to_hf(self, tokenizer: AutoTokenizer) -> DatasetDict:
        """
        Tokenize the dataset using a HuggingFace AutoTokenizer.
        Formats the dataset to torch tensors and converts labels to one-hot.
        :param tokenizer: A HuggingFace AutoTokenizer object.
        :return: Tokenized split datasets ready for the HF Trainer
        """
        tokenized = self.dataset.map(lambda x:
            tokenizer(x['text'], max_length=self.cfg.llm.max_len, padding='max_length', truncation=True),
            batched=True)

        def one_hot_encode_label(example):
            if example['label'] == -1.0:
                example['label'] = [1.0, 0.0]
            elif example['label'] == 1.0:
                example['label'] = [0.0, 1.0]
            return example

        tokenized["train"] = tokenized["train"].map(one_hot_encode_label)
        tokenized["eval"] = tokenized["eval"].map(one_hot_encode_label)

        tokenized["train"].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        tokenized["eval"].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        tokenized["test"].set_format(type='torch', columns=['input_ids', 'attention_mask'])

        return tokenized


if __name__ == "__main__":
    from utils import load_config, set_seed

    cfg = load_config()
    set_seed(cfg.general.seed)
    twitter = TwitterDataset(cfg)
    tokenized_dataset = twitter.tokenize_to_hf(AutoTokenizer.from_pretrained(cfg.hf.model))
