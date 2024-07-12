import os

import numpy as np
from utils import load_config, set_seed
from data_loader import TwitterDataset
from transformers import AutoTokenizer

import torch
from torch import nn, optim
from tqdm import tqdm

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Create a bidirectional LSTM model class
class BiLSTM(nn.Module):
    def __init__(self, cfg, vocab_size):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.num_layers = cfg.model.layers
        self.hidden_size = cfg.model.hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=cfg.model.embedding_dim,
        )
        self.lstm = nn.LSTM(
            cfg.model.embedding_dim, 
            self.hidden_size, 
            self.num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        embeddings = self.embedding(x)
        out, _ = self.lstm(embeddings, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

def process_predictions(y_pred, lower_bound=0.25, upper_bound=0.75):
    """
    Process the predictions to set values <0.25 to 0, >0.75 to 1, and sample in between.
    
    Args:
    y_pred (torch.Tensor): The tensor of predictions between 0 and 1.
    lower_bound (float): The lower bound for certain 0 prediction.
    upper_bound (float): The upper bound for certain 1 prediction.
    
    Returns:
    torch.Tensor: Processed binary predictions.
    """
    
    # Convert the tensor to numpy array for sampling
    y_pred_np = y_pred.cpu().numpy()
    
    # Initialize the output tensor
    processed_preds = torch.zeros_like(y_pred)
    
    # Iterate over the predictions
    for i in range(len(y_pred_np)):
        if y_pred_np[i] < lower_bound:
            processed_preds[i] = 0
        elif y_pred_np[i] > upper_bound:
            processed_preds[i] = 1
        else:
            processed_preds[i] = np.random.binomial(1, y_pred_np[i])
    
    return processed_preds

def eval(model, val_hf, epoch, num_epochs):
    # Evaluate the model on the val set
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        with tqdm(enumerate(val_hf), total=len(val_hf), desc=f"Validation {epoch+1}/{num_epochs}") as pbar:
            for batch, x in pbar:
                x.to(device)
                y_pred = model(x.input_ids).view(-1)
                
                # Calculate predictions
                # predicted = process_predictions(y_pred, lower_bound=0.25, upper_bound=0.75)
                predicted = (y_pred > 0.5).float() 
                
                correct_predictions += (predicted == x.labels).sum().item()
                total_predictions += x.labels.size(0)

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.4f}")

def train(model, train_hf, optimizer, criterion, epoch, num_epochs):
    # Initialize the progress bar for the current epoch
    model.train()
    with tqdm(enumerate(train_hf), total=len(train_hf), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for batch, x in pbar:
            x.to(device)
            y_pred = model(x.input_ids).view(-1)
            loss = criterion(y_pred, x.labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss.item()})

if __name__ == "__main__":
    cfg = load_config()
    set_seed(cfg.general.seed)
    dataset = TwitterDataset(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf.model)
    
    # new tokens
    # new_tokens = ["<user>", "<url>"]
    # new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
    # tokenizer.add_tokens(list(new_tokens))
    
    train_hf, val_hf, test_hf = dataset.to_hf_dataloader(tokenizer)

    model = BiLSTM(cfg, len(tokenizer))
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    epoch_init = -1

    if cfg.model.checkpoint_path:
        checkpoint = torch.load(cfg.model.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_init = checkpoint['epoch']

    num_epochs = cfg.model.epochs
    for epoch in range(epoch_init+1, num_epochs):
        # Train the model on the train set
        train(model, train_hf, optimizer, criterion, epoch, num_epochs)

        # Evaluate the model on the val set
        eval(model, val_hf, epoch, num_epochs)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{cfg.model.out_path}/epoch_{epoch}.ckpt")

