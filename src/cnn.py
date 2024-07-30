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
import torch.nn.functional as F

THRESHOLD = 0.5

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

class TextCNN(nn.Module):
    def __init__(self, cfg, vocab_size):
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, cfg.cnn.embedding_dim)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=cfg.cnn.embedding_dim, 
                      out_channels=cfg.cnn.num_filters, 
                      kernel_size=fs)
            for fs in cfg.cnn.filter_sizes
        ])
        
        self.dropout = nn.Dropout(cfg.cnn.dropout_rate)
        self.fc = nn.Linear(cfg.cnn.num_filters * len(cfg.cnn.filter_sizes), 1)
    
    def forward(self, x):
        
        x = self.embedding(x)  
        x = x.permute(0, 2, 1)  
        
        conv_results = []
        for conv in self.conv_layers:
            conv_x = F.relu(conv(x))  
            pooled_x = F.max_pool1d(conv_x, kernel_size=conv_x.shape[2])  
            conv_results.append(pooled_x)
        
        
        x = torch.cat(conv_results, dim=1)  
        x = x.squeeze(2)  
        x = self.dropout(x)
        logits = self.fc(x)  
        
        return logits

    def run_test(self, test_dl):
        
        self.eval()  
        predictions = np.array([])
        with torch.no_grad():  
            with tqdm(enumerate(test_dl), total=len(test_dl), desc=f"Testing") as pbar:
                for _, batch in pbar:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    y_pred = self(batch['input_ids']).view(-1)
                    predicted = (torch.sigmoid(y_pred) > THRESHOLD).float()
                    
                    predictions = np.concatenate((predictions, predicted.cpu()))

        return predictions

    def run_eval(self, val_dl, epoch, num_epochs):
        
        self.eval()  
        predictions = np.array([])
        labels = np.array([])
        with torch.no_grad():  
            with tqdm(enumerate(val_dl), total=len(val_dl), desc=f"Validation {epoch+1}/{num_epochs}") as pbar:
                for _, batch in pbar:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    y_pred = self(batch['input_ids']).view(-1)
                    
                    predictions = np.concatenate((predictions, torch.sigmoid(y_pred).cpu()))
                    labels = np.concatenate((labels, batch['labels'].cpu()))

        
        metrics = compute_metrics((predictions, labels))
        print(f"Epoch {epoch}/{num_epochs}, Val Accuracy: {metrics['accuracy']:.4f}, Val F1: {metrics['f1']:.4f}")
        return metrics

    def run_train(self, train_dl, optimizer, criterion, epoch, num_epochs):
        
        self.train()
        with tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch}/{num_epochs}") as pbar:
            for _, batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                y_pred = self(batch['input_ids']).view(-1)
                loss = criterion(y_pred, batch['labels'])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': loss.item()})

if __name__ == "__main__":
    cfg = load_config()
    set_seed(cfg.general.seed)

    twitter = TwitterDataset(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.model)
    tokenized_dataset = twitter.tokenize_to_hf(tokenizer)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dl = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=cfg.cnn.batch_size, collate_fn=data_collator)
    eval_dl = DataLoader(tokenized_dataset['eval'], shuffle=True, batch_size=cfg.cnn.batch_size, collate_fn=data_collator)
    test_dl = DataLoader(tokenized_dataset['test'], shuffle=False, batch_size=cfg.cnn.batch_size, collate_fn=data_collator)

    model = TextCNN(cfg, len(tokenizer))
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    epoch_init = -1

    if cfg.cnn.resume_from_checkpoint:
        checkpoint = torch.load(cfg.cnn.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_init = checkpoint['epoch']

    max_acc = 0
    num_epochs = cfg.cnn.epochs
    for epoch in range(epoch_init+1, num_epochs):        
        model.run_train(train_dl, optimizer, criterion, epoch, num_epochs)
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
