from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from box import Box
from datasets import Dataset
import numpy as np 
import time

class RFClassifier: 
    def __init__(self, config: Box): 
        self.cfg = config
        self.model = RandomForestClassifier(
            n_estimators=self.cfg.rf.n_estimators,
            max_depth=self.cfg.rf.max_depth, 
            max_features=self.cfg.rf.max_features,
            random_state=self.cfg.general.seed,
            n_jobs=-1,
            verbose=3)
    
    def train(self, train_data: Dataset) -> None: 
        self.model.fit(train_data["features"], train_data["label"])

    def test(self, test_data: Dataset) -> np.ndarray: 
        return self.model.predict(test_data["features"])
