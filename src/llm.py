import os.path

from box import Box
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import DatasetDict
from transformers import (
    AutoTokenizer, AutoModel, DataCollatorWithPadding,
    Trainer, TrainingArguments, EarlyStoppingCallback,)
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model
import evaluate


THRESHOLD = 0.5


def preprocessor(tweet: str) -> str:
    """Custom preprocessor for the Bertweet Tokenizer."""
    return tweet.replace("<user>", "@USER").replace("<url>", "http")


def compute_metrics(eval_pred):
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")
    predictions, labels = eval_pred

    predictions = (predictions >= THRESHOLD).astype(int)
    labels = labels.astype(int)

    return {
        "accuracy": load_accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": load_f1.compute(predictions=predictions, references=labels)["f1"]
    }


class CustomRobertaForSequenceClassification(nn.Module):
    """Custom RobertaForSequenceClassification without dense layer in classifier."""

    def __init__(self, original_model):
        super(CustomRobertaForSequenceClassification, self).__init__()
        self.roberta = original_model
        self.classifier_dropout = nn.Dropout(original_model.config.hidden_dropout_prob)
        self.classifier = nn.Linear(original_model.config.hidden_size, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0][:, 0, :] # [CLS] token output
        logits = self.classifier_dropout(sequence_output)
        logits = self.classifier(logits).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class LLMClassifier():
    """
    Sentiment classifier using a Large Language Model with a classification head.
    Initializes the model in cfg.llm.model, which can be a Hub model or a local checkpoint.
    """
    def __init__(self, config: Box):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.llm.model)
        original_model = AutoModel.from_pretrained(self.cfg.llm.model)
        self.model = CustomRobertaForSequenceClassification(original_model)

        if self.cfg.llm.use_lora:
            lora_config = LoraConfig(
                r=self.cfg.llm.lora_r,
                lora_alpha=self.cfg.llm.lora_alpha,
                use_rslora=True,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["classifier"],
                task_type=TaskType.SEQ_CLS,
            )

            self.model = get_peft_model(self.model, lora_config)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        precision = "fp16" if self.cfg.llm.use_fp16 else "fp32"
        print(f"\n'{self.cfg.llm.model}' loaded with {trainable} trainable {precision} parameters.\n")


    def train(self, dataset: DatasetDict):
        """
        Train the model using the HuggingFace Trainer API.
        :param dataset: Tokenized split datasets ready for the HF Trainer
        """
        training_args = TrainingArguments(
            output_dir=os.path.join(self.cfg.data.checkpoint_path, self.cfg.general.run_id),
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=self.cfg.llm.epochs,
            learning_rate=self.cfg.llm.lr,
            warmup_steps=self.cfg.llm.warmup_steps,
            per_device_train_batch_size=self.cfg.llm.batch_size,
            per_device_eval_batch_size=self.cfg.llm.batch_size,
            gradient_accumulation_steps=self.cfg.llm.gradient_accumulation_steps,
            weight_decay=self.cfg.llm.weight_decay,
            fp16=self.cfg.llm.use_fp16,
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
            sigmoid = nn.Sigmoid()
            for batch in tqdm(loader, desc="Computing Submission"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)[0]
                outputs = sigmoid(outputs).cpu().detach().numpy()

                result = np.where(outputs >= THRESHOLD, 1, -1).reshape(-1, 1)
                results.append(result)

        return np.squeeze(np.vstack(results))


if __name__ == "__main__":
    from utils import load_config, set_seed, save_outputs
    from data_loader import TwitterDataset

    cfg = load_config()
    set_seed(cfg.general.seed)

    llm = LLMClassifier(cfg)
    twitter = TwitterDataset(cfg, preprocessor)
    tokenized_dataset = twitter.tokenize_to_hf(llm.tokenizer)
    llm.train(tokenized_dataset)

    llm_outputs = llm.test(tokenized_dataset)
    save_outputs(llm_outputs, cfg.general.run_id)
