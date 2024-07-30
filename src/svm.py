from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from box import Box
from datasets import Dataset
import numpy as np 
import time

def preprocessor(tweet: str) -> str:
    return tweet.replace("<user>", "@USER")

class SVMClassifier: 
    def __init__(self, config: Box): 
        self.cfg = config
        self.model = SVC(
            kernel=config.svm.kernel,
            C=config.svm.C,
            class_weight=config.svm.class_weight,
            random_state=self.cfg.general.seed,
            verbose=True) 
    
    def train(self, train_data: Dataset) -> None: 
        self.model.fit(train_data["features"], train_data["label"])

    def test(self, test_data: Dataset) -> np.ndarray: 
        return self.model.predict(test_data["features"])