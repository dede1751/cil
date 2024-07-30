import torch
from torch import nn
import torch.nn.functional as F

from neural_network import NeuralNetwork

class TextCNN(NeuralNetwork):
    def __init__(self, cfg, vocab_size, device):
        super(TextCNN, self).__init__(device)
        
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