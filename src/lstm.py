import torch
from torch import nn
from neural_network import NeuralNetwork

class LSTM(NeuralNetwork):
    def __init__(self, cfg, vocab_size, device):
        super(LSTM, self).__init__(device)
        self.cfg = cfg
        self.device = device
        self.vocab_size = vocab_size
        self.num_layers = self.cfg.lstm.layers
        self.hidden_size = self.cfg.lstm.hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.cfg.lstm.embedding_dim,
        )
        self.lstm = nn.LSTM(
            self.cfg.lstm.embedding_dim, 
            self.hidden_size, 
            self.num_layers, 
            batch_first=True, 
            bidirectional=self.cfg.lstm.bidirectional,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * (2 if self.cfg.lstm.bidirectional else 1) , 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        m = 2 if self.cfg.lstm.bidirectional else 1
        h0 = torch.zeros(self.num_layers * m, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * m, x.size(0), self.hidden_size).to(self.device)

        embeddings = self.embedding(x)
        out, _ = self.lstm(embeddings, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out