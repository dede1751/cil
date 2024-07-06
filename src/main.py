import os

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


if __name__ == "__main__":
    cfg = load_config()
    set_seed(cfg.general.seed)
    dataset = TwitterDataset(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf.model)
    train_hf, val_hf, test_hf = dataset.to_hf_dataloader(tokenizer)

    model = BiLSTM(cfg, len(tokenizer))
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    epoch_init = -1

    # checkpoint = torch.load("./checkpoints/best.ckpt", map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch_init = checkpoint['epoch']

    num_epochs = cfg.model.epochs
    for epoch in range(epoch_init+1, num_epochs):
        model.train()
        # Initialize the progress bar for the current epoch
        with tqdm(enumerate(train_hf), total=len(train_hf), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch, x in pbar:
                x.to(device)
                y_pred = model(x.input_ids).view(-1)
                loss = criterion(y_pred, x.labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': loss.item()})

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
                    predicted = (y_pred > 0.5).float()  # Assuming binary classification with a threshold of 0.5
                    correct_predictions += (predicted == x.labels).sum().item()
                    total_predictions += x.labels.size(0)

        # Calculate accuracy
        accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.4f}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"./checkpoints/epoch_{epoch}.ckpt")

