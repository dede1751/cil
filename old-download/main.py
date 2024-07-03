# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset, load_metric
import pandas as pd
import numpy as np
from dataset import df

tokenizer = AutoTokenizer.from_pretrained("Twitter/twhin-bert-base")

dataset = Dataset.from_pandas(df, split=['train', 'test']).shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.3)

def preprocess_function(examples):
   return tokenizer(examples["x"], truncation=True)
 
tokenized_train = dataset['train'].map(preprocess_function, batched=True)
tokenized_test = dataset['test'].map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("Twitter/twhin-bert-base")
 
def compute_metrics(eval_pred):
  load_accuracy = load_metric("accuracy")
  load_f1 = load_metric("f1")

  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
  f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
  return {"accuracy": accuracy, "f1": f1}
  
training_args = TrainingArguments(
   output_dir='demo_run',
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch",
)
 
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

trainer.train()