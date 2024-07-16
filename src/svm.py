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
            random_state=self.cfg.general.seed) 
    
    def train(self, train_data: Dataset) -> None: 
        self.model.fit(train_data["features"], train_data["label"])

    def test(self, test_data: Dataset) -> np.ndarray: 
        return self.model.predict(test_data["features"])
        
if __name__ == "__main__":
    from utils import load_config, set_seed, save_outputs
    from data_loader import TwitterDataset

    cfg = load_config()
    set_seed(cfg.general.seed)

    svm = SVMClassifier(cfg)
    twitter = TwitterDataset(cfg, preprocessor)
    vectorizer = TfidfVectorizer(max_features=cfg.data.sparse_max_features)

    vectorized_dataset = twitter.vectorize(vectorizer)

    print("Training...")
    start = time.time()
    svm.train(vectorized_dataset["train"])
    end = time.time()
    print(f"Finished Training in {end - start} seconds.")
    
    svm_outputs_validation = svm.test(vectorized_dataset["eval"])

    report_train = classification_report(vectorized_dataset["train"]["label"], svm.test(vectorized_dataset["train"]))
    report_val = classification_report(vectorized_dataset["eval"]["label"], svm_outputs_validation)
    
    print("Training report:\n", report_train)
    print("\n")
    print("Validation report:\n", report_val)

    svm_outputs = svm.test(vectorized_dataset["test"])
    svm_outputs = np.where(svm_outputs == 0, -1, 1)
    save_outputs(svm_outputs, cfg.general.run_id)
