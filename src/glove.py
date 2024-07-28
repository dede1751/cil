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

class GloveTokenizer:
    def __init__(self, embedding, tokenizer): 
        self.embedding = embedding
        self.tokenizer = tokenizer
    
    def __call__(self, data):
        input_ids = [self.embedding.get_tokens_id(self.tokenizer.tokenize(tweet)) for tweet in data['text']] 
        return {'input_ids' : input_ids}

def preprocessor(tweet: str) -> str:
    """Taken from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb"""
    import re
    import wordsegment
    
    tweet = re.sub(r'[+-]?\d([:.,]?\d)*', " <number> ", tweet)

    def hashtag_split(match):
        hashtag_body = match.group(0)[1:]
        new_body = hashtag_body
        try:
            new_body = " ".join(wordsegment.segment(hashtag_body))
        finally:
            if hashtag_body.upper() == hashtag_body:   
                new_body += "<allcaps>"
            return f"<hashtag> {new_body}"
    
    tweet = re.sub(r"#\S+", hashtag_split, tweet)
    
    # Replace sequences of "!" with "! <repeat>"
    tweet = re.sub(r"(?:\s*!){2,}", r" ! <repeat>", tweet)
    # Replace sequences of "?" with "? <repeat>"
    tweet = re.sub(r"(?:\s*\?){2,}", r"? <repeat>", tweet)
    # Replace sequences of "." with ". <repeat>"
    tweet = re.sub(r"(?:\s*\.){2,}", r" . <repeat>", tweet)
    
    # Replace elongated words with "<elong>"
    tweet = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>", tweet)
    # Replace all-caps words with "<allcaps>"
    tweet = re.sub(r"\b([A-Z]{2,})\b", lambda m: f"{m.group(0).lower()} <allcaps>", tweet)

    return tweet.lower()

