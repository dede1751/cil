from box import Box
import numpy as np
import torch 
import torch.nn as nn
import codecs

class GloveEmbedding: 
    def __init__(self, cfg: Box): 
        self.cfg = cfg
        self.unknown_token = "<unknown>"
        self._load_glove(f"./{self.cfg.embedding.folder}/{self.cfg.embedding.name}.{self.cfg.embedding.dim}d.txt")
    
    def _load_glove(self, path: str) -> None: 
        idx = 1
        self.vocab = {}
        self.embeddings = [np.zeros(self.cfg.embedding.dim, dtype=np.float32)]
        with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                values = line.split()
                if len(values) >= 1:
                    word = values[0]
                    try:
                        vector = np.array(values[1:], dtype=np.float32)
                        if vector.shape[0] == self.cfg.embedding.dim:  # Ensure correct dimension
                            self.vocab[word] = idx
                            self.embeddings.append(vector)
                            idx += 1
                        else:
                            print(f"Skipping line with incorrect dimensions: {line[:50]}...")
                    except ValueError as e:
                        print(f"Skipping line due to parsing error: {line[:50]}... Error: {e}")
    
    def get_token_id(self, tokens: list[str]) -> list[np.ndarray]:
        token_ids = [self.vocab[token] if token in self.vocab else self.vocab[self.unknown_token] for token in tokens]
        token_ids += [0] * (self.cfg.data.max_length - len(token_ids))
        return token_ids
    
    def get_embedding(self) -> nn.Embedding:    
        embedding_tensor = torch.from_numpy(np.array(self.embeddings))
        return nn.Embedding.from_pretrained(
            embedding_tensor, 
            padding_idx=0,
            freeze=self.cfg.embedding.freeze)