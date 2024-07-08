import os

from box import Box
import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
    Trainer, TrainingArguments, EarlyStoppingCallback,)
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model
from scipy.special import softmax
import evaluate


def compute_metrics(eval_pred):
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")
    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=-1)
    labels = np.argmax(labels, axis=-1)

    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}

def preprocess_twitter_roberta(text):
    """
    Tweet pre-processing function for 'twitter-roberta-base-sentiment-latest'.
    https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    """
    return text.replace("<user>", "@user").replace("<url>", "http")


class LLMClassifier():
    """
    Sentiment classifier using a Large Language Model with a classification head.
    Initializes the model in cfg.llm.model, which can be a Hub model or a local checkpoint.
    """
    def __init__(self, config: Box):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.llm.model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.llm.model,
            num_labels=2,
            id2label={0: "NEGATIVE", 1: "POSITIVE"},
            label2id={"NEGATIVE": 0, "POSITIVE": 1},
            ignore_mismatched_sizes=True
        )

        if self.cfg.llm.special_tokens:
            special_tokens_dict = {'additional_special_tokens': ['<user>', '<url>']}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.cfg.llm.use_lora:
            lora_config = LoraConfig(
                r=self.cfg.llm.lora_r,
                lora_alpha=self.cfg.llm.lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["classifier"],
                task_type=TaskType.SEQ_CLS,
            )

            self.model = get_peft_model(self.model, lora_config)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n'{self.cfg.llm.model}' loaded with {trainable} trainable parameters.\n")

    def train(self, dataset: DatasetDict):
        """
        Train the model using the HuggingFace Trainer API.
        :param dataset: Tokenized split datasets ready for the HF Trainer
        """
        training_args = TrainingArguments(
            output_dir=os.path.join(self.cfg.data.checkpoint_path, self.cfg.general.run_id),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.cfg.llm.lr,
            per_device_train_batch_size=self.cfg.llm.batch_size,
            per_device_eval_batch_size=self.cfg.llm.batch_size,
            gradient_accumulation_steps=self.cfg.llm.gradient_accumulation_steps,
            num_train_epochs=self.cfg.llm.epochs,
            weight_decay=self.cfg.llm.weight_decay,
            metric_for_best_model='accuracy',
            greater_is_better=True,
            load_best_model_at_end=True,
        )

        data_collator = DataCollatorWithPadding(self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        trainer.train()

    def test(self, dataset: DatasetDict) -> np.ndarray:
        """
        Run the model on the test dataset.
        :param dataset: Tokenized split datasets ready for the HF Trainer
        :return: Test dataset output labels (-1 for negative, 1 for positive)
        """
        self.model.to(self.device)
        self.model.eval()

        loader = DataLoader(dataset["test"], batch_size=self.cfg.llm.batch_size, shuffle=False)
        results = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Computing Submission"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                scores = outputs.logits.cpu().detach().numpy()

                if self.cfg.llm.sample_from_output:
                    scores = softmax(scores, axis=1)
                    result = np.array(
                        [np.random.choice([-1, 1], p=score) for score in scores]).reshape(-1, 1)
                else:
                    result = np.where(scores[:, 0] > scores[:, 1], -1, 1).reshape(-1, 1)

                results.append(result)

        return np.squeeze(np.vstack(results))


if __name__ == "__main__":
    from utils import load_config, set_seed, save_outputs
    from data_loader import TwitterDataset

    cfg = load_config()
    set_seed(cfg.general.seed)

    llm = LLMClassifier(cfg)
    twitter = TwitterDataset(cfg)
    tokenized_dataset = twitter.tokenize_to_hf(llm.tokenizer, preprocess_twitter_roberta)
    llm.train(tokenized_dataset)

    llm_outputs = llm.test(tokenized_dataset)
    save_outputs(llm_outputs, cfg.general.run_id)
