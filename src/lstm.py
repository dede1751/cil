import os

import numpy as np
from utils import load_config, set_seed, save_outputs
from data_loader import TwitterDataset
from transformers import DataCollatorWithPadding, AutoTokenizer
import torch
from torch import nn, optim
from tqdm import tqdm
import evaluate
from torch.utils.data import DataLoader
from llm import compute_metrics

THRESHOLD = 0.5

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Create a bidirectional LSTM model class
class LSTM(nn.Module):
    def __init__(self, cfg, vocab_size):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.num_layers = cfg.lstm.layers
        self.hidden_size = cfg.lstm.hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=cfg.lstm.embedding_dim,
        )
        self.lstm = nn.LSTM(
            cfg.lstm.embedding_dim, 
            self.hidden_size, 
            self.num_layers, 
            batch_first=True, 
            bidirectional=cfg.lstm.bidirectional,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * (2 if cfg.lstm.bidirectional else 1) , 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        m = 2 if cfg.lstm.bidirectional else 1
        h0 = torch.zeros(self.num_layers * m, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * m, x.size(0), self.hidden_size).to(device)

        embeddings = self.embedding(x)
        out, _ = self.lstm(embeddings, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

    def run_test(self, test_dl):
        # Evaluate the model on the val set
        self.eval()  # Set the model to evaluation mode
        predictions = np.array([])
        with torch.no_grad():  # Disable gradient calculation for evaluation
            with tqdm(enumerate(test_dl), total=len(test_dl), desc=f"Testing") as pbar:
                for batch, x in pbar:
                    x.to(device)
                    y_pred = self(x.input_ids).view(-1)
                    predicted = (y_pred > 0.5).float() 
                    
                    predictions = np.concatenate((predictions, predicted.cpu()))

        return predictions


    def run_eval(self, val_dl, epoch, num_epochs):
        # Evaluate the model on the val set
        self.eval()  # Set the model to evaluation mode
        predictions = np.array([])
        labels = np.array([])
        with torch.no_grad():  # Disable gradient calculation for evaluation
            with tqdm(enumerate(val_dl), total=len(val_dl), desc=f"Validation {epoch+1}/{num_epochs}") as pbar:
                for batch, x in pbar:
                    x.to(device)
                    y_pred = self(x.input_ids).view(-1)
                    
                    predictions = np.concatenate((predictions, y_pred.cpu()))
                    labels = np.concatenate((labels, x.labels.cpu()))

        # Calculate accuracy
        metrics = compute_metrics((predictions, labels))
        print(f"Epoch {epoch}/{num_epochs}, Val Accuracy: {metrics['accuracy']:.4f}, Val F1: {metrics['f1']:.4f}")
        return metrics

    def run_train(self, train_dl, optimizer, criterion, epoch, num_epochs):
        # Initialize the progress bar for the current epoch
        self.train()
        with tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch}/{num_epochs}") as pbar:
            for batch, x in pbar:
                x.to(device)
                y_pred = self(x.input_ids).view(-1)
                loss = criterion(y_pred, x.labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': loss.item()})

if __name__ == "__main__":
    cfg = load_config()
    set_seed(cfg.general.seed)

    twitter = TwitterDataset(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.model)
    tokenized_dataset = twitter.tokenize_to_hf(tokenizer, padding=False)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dl = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=cfg.lstm.batch_size, collate_fn=data_collator)
    eval_dl = DataLoader(tokenized_dataset['eval'], shuffle=True, batch_size=cfg.lstm.batch_size, collate_fn=data_collator)
    test_dl = DataLoader(tokenized_dataset['test'], shuffle=False, batch_size=cfg.lstm.batch_size, collate_fn=data_collator)

    model = LSTM(cfg, len(tokenizer))
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    epoch_init = -1

    if cfg.lstm.resume_from_checkpoint:
        checkpoint = torch.load(cfg.lstm.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_init = checkpoint['epoch']

    max_acc = 0
    num_epochs = cfg.lstm.epochs
    for epoch in range(epoch_init+1, num_epochs):
        # Train the model on the train set
        model.run_train(train_dl, optimizer, criterion, epoch, num_epochs)

        # Evaluate the model on the val set
        acc = model.run_eval(eval_dl, epoch, num_epochs)['accuracy']

        if acc >= max_acc:
            max_acc = acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{cfg.data.checkpoint_path}/best.ckpt")
            
            outputs = model.run_test(test_dl)
            save_outputs(outputs, cfg.general.run_id)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{cfg.data.checkpoint_path}/final.ckpt")

