from torch import nn 
import torch.nn.functional as F

from neural_network import NeuralNetwork

class FFNN(NeuralNetwork):
    def __init__(self, cfg, vocab_size, device):
        super(FFNN, self).__init__(device)
        
        self.embedding = nn.Embedding(vocab_size, cfg.ffnn.embedding_dim)
        self.fc1 = nn.Linear(cfg.ffnn.embedding_dim, cfg.ffnn.hidden_dim1)
        self.fc2 = nn.Linear(cfg.ffnn.hidden_dim1, cfg.ffnn.hidden_dim2)
        self.fc3 = nn.Linear(cfg.ffnn.hidden_dim2, 1)
        self.dropout = nn.Dropout(cfg.ffnn.dropout_rate)
    
    def forward(self, x):
        
        x = self.embedding(x)
        x = x.mean(dim=1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits