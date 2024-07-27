from box import Box
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn

class GloveEmbedding: 
    def __init__(self, cfg: Box): 
        self.cfg = cfg
        self.unknown_token = "<unknown>"
        self._load_glove(f"./{self.cfg.embedding.folder}/{self.cfg.embedding.name}.{self.cfg.embedding.dim}d.txt")
    
    def _load_glove(self, path: str) -> None: 
        idx = 1
        self.vocab = {}
        self.embeddings = [np.zeros(self.cfg.embedding.dim, dtype=np.float32)]

        df = pd.read_csv(path, sep=" ", quoting=3, header=None, index_col=0, dtype={0: str})

        for word, vector in df.iterrows():
            self.vocab[word] = idx
            self.embeddings.append(vector.values.astype(np.float32))
            idx += 1
    
    def get_tokens_id(self, tokens: list[str]) -> list[np.ndarray]:
        token_ids = [self.vocab.get(token, self.vocab[self.unknown_token]) for token in tokens]
        token_ids += [0] * (self.cfg.data.max_length - len(token_ids))
        return token_ids
    
    def get_embedding(self) -> nn.Embedding:    
        embedding_tensor = torch.from_numpy(np.array(self.embeddings))
        return nn.Embedding.from_pretrained(
            embedding_tensor, 
            padding_idx=0,
            freeze=self.cfg.embedding.freeze)
    
if __name__ == "__main__": 
    from nltk import TweetTokenizer

    tweet = "<user> shucks well i work all week so now i can't come cheer you on ! oh and put those batteries in your calculator ! ! !"
    cfg = Box({
        "embedding": {
            "folder": "pretrained_embeddings",
            "name": "glove",
            "dim": 200,
            "freeze": True
        },
        "data": {
            "max_length": 128
        }
    })

    print("tweet:", tweet)

    glove = GloveEmbedding(cfg)
    tokenizer = TweetTokenizer()

    tokens = tokenizer.tokenize(tweet)
    print("tokens:", tokens)

    token_ids = glove.get_token_id(tokens)  
    print("token_ids:", token_ids)  

