from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from box import Box
from datasets import Dataset
import numpy as np 

class LogitClassifier: 
    def __init__(self, config: Box): 
        self.cfg = config
        self.model = LogisticRegression(
            max_iter=config.logit.max_iter,
            random_state=self.cfg.general.seed,
            verbose=1,
            n_jobs=-1) 
    
    def train(self, train_data: Dataset) -> None: 
        self.model.fit(train_data["features"], train_data["label"])

    def test(self, test_data: Dataset) -> np.ndarray: 
        return self.model.predict(test_data["features"])
