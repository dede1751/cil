from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from box import Box
from datasets import Dataset
import numpy as np 
import time

def preprocessor(tweet: str) -> str:
    return tweet.replace("<user>", "@USER")

class RFClassifier: 
    def __init__(self, config: Box): 
        self.cfg = config
        self.model = RandomForestClassifier(
            n_estimators=self.cfg.rf.n_estimators,
            max_depth=self.cfg.rf.max_depth, 
            max_features=self.cfg.rf.max_features,
            random_state=self.cfg.general.seed,
            n_jobs=-1)
    
    def train(self, train_data: Dataset) -> None: 
        self.model.fit(train_data["features"], train_data["label"])

    def test(self, test_data: Dataset) -> np.ndarray: 
        return self.model.predict(test_data["features"])
        
if __name__ == "__main__":
    from utils import load_config, set_seed, save_outputs
    from data_loader import TwitterDataset

    cfg = load_config()
    set_seed(cfg.general.seed)

    logit = RFClassifier(cfg)
    twitter = TwitterDataset(cfg, preprocessor)
    vectorizer = TfidfVectorizer(max_features=cfg.data.sparse_max_features)

    vectorized_dataset = twitter.vectorize(vectorizer)

    print("Training...")
    start = time.time()
    logit.train(vectorized_dataset["train"])
    end = time.time()
    print(f"Finished Training in {end - start} seconds.")

    logit_outputs_validation = logit.test(vectorized_dataset["eval"])

    report_train = classification_report(vectorized_dataset["train"]["label"], logit.test(vectorized_dataset["train"]))
    report_val = classification_report(vectorized_dataset["eval"]["label"], logit_outputs_validation)
    
    print("Training report:\n", report_train)
    print("\n")
    print("Validation report:\n", report_val)

    logit_outputs = logit.test(vectorized_dataset["test"])
    logit_outputs = np.where(logit_outputs == 0, -1, 1)
    save_outputs(logit_outputs, cfg.general.run_id)
