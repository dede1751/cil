import wordsegment
import glove
import nltk
import numpy as np
import torch
import time

from utils import load_config, set_seed, save_outputs
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data_loader import TwitterDataset

import logit, rf, svm

if __name__ == "__main__":
    cfg = load_config()
    set_seed(cfg.general.seed)
    wordsegment.load()

    preprocessor = None
    if cfg.baselines.data == "glove": 
        preprocessor = glove.preprocessor

    twitter = TwitterDataset(cfg, preprocessor)

    if cfg.baselines.data == "count": 
        vectorizer = CountVectorizer(max_features=cfg.data.sparse_max_features)
        dataset = twitter.vectorize(vectorizer)
    elif cfg.baselines.data == "tfidf":
        vectorizer = TfidfVectorizer(max_features=cfg.data.sparse_max_features)
        dataset = twitter.vectorize(vectorizer)
    elif cfg.baselines.data == "glove":
        embedding = glove.GloveEmbedding(cfg)
        tokenizer = glove.GloveTokenizer(embedding, nltk.TweetTokenizer())
        dataset = twitter.tokenize(tokenizer)

        emebdding_matrix = embedding.get_embedding()
        dataset = dataset.map(lambda x: 
                    {"features": [torch.mean(emebdding_matrix(input_ids), axis=0).squeeze() for input_ids in x["input_ids"]]}, 
                    batched=True)
        dataset["train"].set_format(type='numpy', columns=['features', 'label'])
        dataset["eval"].set_format(type='numpy', columns=['features', 'label'])
        dataset["test"].set_format(type='numpy', columns=['features'])
    else:
        raise NotImplementedError

    if cfg.baselines.model == "logit":
        model = logit.LogitClassifier(cfg)
    elif cfg.baselines.model == "rf": 
        model = rf.RFClassifier(cfg)
    elif cfg.baselines.model == "svm": 
        model = svm.SVMClassifier(cfg)
    else: 
        raise NotImplementedError
    
    print("Training...")
    start = time.time()
    model.train(dataset["train"])
    end = time.time()
    print(f"Finished Training in {end - start} seconds.")
    
    eval_outputs = model.test(dataset["eval"])
    eval_true = dataset["eval"]["label"]

    print(f"Accuracy: {accuracy_score(eval_true, eval_outputs)}")
    print(f"F1: {f1_score(eval_true, eval_outputs)}")   
    print(f"Precision: {precision_score(eval_true, eval_outputs)}")
    print(f"Recall: {recall_score(eval_true, eval_outputs)}")
    
    test_outputs = model.test(dataset["test"])
    test_outputs = np.where(test_outputs == 0, -1, 1)

    save_outputs(test_outputs, cfg.general.run_id)