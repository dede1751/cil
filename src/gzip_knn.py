import gzip
import numpy as np
import torch
import evaluate
import os
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from utils import load_config, set_seed, save_outputs
from data_loader import TwitterDataset
from transformers import DataCollatorWithPadding, AutoTokenizer
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from llm import compute_metrics

class gzipKNN:
    def __init__(self):
        """
        Initializes the TweetClassifier with the number of nearest neighbors to consider.
        
        :param k: The number of nearest neighbors to use for classification.
        """
        self.k = cfg.gzipknn.k
        
        
    def _compress_length(self, text):
        """
        Helper method to get the compressed length of a text.
        
        :param text: Input text (string).
        :return: Compressed length of the text.
        """
        return len(gzip.compress(text.encode()))

    def predict(self, test_texts, training_texts):
        """
        Predicts the sentiment of tweets in the test set.
        
        :param test_texts: List of tweets (str) to be classified.
        :return: List of predicted class labels for the test set.
        """
        predictions = []
        training_labels = np.array(training_texts["label"])
        compressed_train =  [(x2["text"],self._compress_length(x2["text"])) for x2 in training_texts]
        
        for x1 in test_texts:
            x1 = x1["text"]
            Cx1 = self._compress_length(x1)
            
            distances = []

            for x2, Cx2 in compressed_train:

                x1x2 = " ".join([x1, x2])
                Cx1x2 = self._compress_length(x1x2)
                ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
                distances.append(ncd)

            sorted_idx = np.argsort(distances)
            top_k_labels = list(training_labels[sorted_idx[:self.k]])
            predicted_label = max(set(top_k_labels), key=top_k_labels.count)
            predictions.append(predicted_label)

        return predictions

if __name__ == "__main__":
    cfg = load_config()
    set_seed(cfg.general.seed)

    twitter = TwitterDataset(cfg)
    
    dataset = twitter._load_dataset()
    
    train = dataset["train"].select(range(0,2000000,5))
    
    test = dataset["test"]
    
    model = gzipKNN()
    
    predictions = model.predict(test, train)
    
    predictions = np.where(predictions == 0, -1, 1)
    save_outputs(np.array(predictions), cfg.general.run_id)

    
