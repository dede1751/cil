import os

from datasets import DatasetDict, Dataset
import pandas as pd
from box import Box
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer
from sklearn.model_selection import train_test_split

class TwitterDataset():
    """
    Custom dataset class for Twitter data. Provides datasets for the different ML frameworks.
    """
    def __init__(self, config: Box):
        self.cfg = config
        self._load_data()

    def _read_data(self, file: str) -> pd.DataFrame:
        """
        Reads the data from the given data file.
        """
        file_path = os.path.join(self.cfg.data.path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        tweets = [line.strip() for line in lines]
        return pd.DataFrame(tweets, columns=['tweet'])

    def _load_data(self):
        if self.cfg.data.use_full_dataset:
            train_neg = self._read_data("train_neg_full.txt")
            train_pos = self._read_data("train_pos_full.txt")
        else:
            train_neg = self._read_data("train_neg.txt")
            train_pos = self._read_data("train_pos.txt")

        train_neg['label'] = 0.0
        train_pos['label'] = 1.0
        train_df = pd.concat([train_neg, train_pos], ignore_index=True)

        test_df = self._read_data("test_data.txt")
        test_df['tweet'] = test_df['tweet'].apply(lambda x: ",".join(x.split(",", 1)[1:]))

        train_df, val_df = train_test_split(train_df, test_size=0.2)

        # Remove duplicates, keeping the most common label
        most_common_labels = train_df.groupby('tweet')['label'].agg(lambda x: x.value_counts().idxmax())
        train_df = train_df.drop_duplicates(subset='tweet').set_index('tweet')
        train_df['label'] = most_common_labels

        self.dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'eval': Dataset.from_pandas(val_df),
            'test': Dataset.from_pandas(test_df)
        })

    def to_hf_dataloader(self, tokenizer: AutoTokenizer) -> tuple[DataLoader, DataLoader]:
        """
        Tokenize the dataset using a HuggingFace AutoTokenizer.
        :param tokenizer: A HuggingFace AutoTokenizer object.
        :return: A tuple of two DataLoaders for training and testing.
        """
        tokenized = self.dataset.map(lambda x:
            tokenizer(x['tweet'], max_length=self.cfg.hf.max_len, padding='max_length', truncation=True),
            batched=True)

        tokenized['train'].set_format('torch', columns=["input_ids", "attention_mask", "label"])
        tokenized['eval'].set_format('torch', columns=["input_ids", "attention_mask", "label"])
        tokenized['test'].set_format('torch', columns=["input_ids", "attention_mask"])

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        train = DataLoader(tokenized['train'], shuffle=True, batch_size=self.cfg.hf.train_batch, collate_fn=data_collator)
        eval = DataLoader(tokenized['eval'], shuffle=True, batch_size=self.cfg.hf.train_batch, collate_fn=data_collator)
        test = DataLoader(tokenized['test'], batch_size = self.cfg.hf.test_batch, collate_fn=data_collator)
        return train, eval, test


if __name__ == "__main__":
    from utils import load_config, set_seed

    cfg = load_config()
    set_seed(cfg.general.seed)
    dataset = TwitterDataset(cfg)
    train_hf, val_hf, test_hf = dataset.to_hf_dataloader(AutoTokenizer.from_pretrained(cfg.hf.model))
