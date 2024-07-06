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
import evaluate


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)
    labels = [1.0 if l[0] == 0 else 0.0 for l in labels]

    return accuracy.compute(predictions=predictions, references=labels)


class LLMClassifier():
    """
    Sentiment classifier using a Large Language Model with a classification head.
    """
    def __init__(self, config: Box, checkpoint_path: str = None):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if checkpoint_path is None:
            model_name = self.cfg.llm.model
        else:
            model_name = checkpoint_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
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
            output_dir=self.cfg.data.checkpoint_path,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.cfg.llm.lr,
            per_device_train_batch_size=self.cfg.llm.batch_size,
            per_device_eval_batch_size=self.cfg.llm.batch_size,
            num_train_epochs=self.cfg.llm.epochs,
            weight_decay=self.cfg.llm.weight_decay,
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
        results = trainer.evaluate()
        print(results)

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
                result = np.where(scores[:, 0] > scores[:, 1], -1, 1).reshape(-1, 1)
                results.append(result)

        return np.squeeze(np.vstack(results))


if __name__ == "__main__":
    from utils import load_config, set_seed, save_outputs
    from data_loader import TwitterDataset

    cfg = load_config()
    set_seed(cfg.general.seed)

    llm = LLMClassifier(cfg)
    tokenized_dataset = TwitterDataset(cfg).tokenize_to_hf(llm.tokenizer)
    llm.train(tokenized_dataset)

    outputs = llm.test(tokenized_dataset)
    save_outputs(outputs, "llm_outputs")
