import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import wordsegment

THRESHOLD = 0.5

def preprocessor(tweet: str) -> str:
    """Taken from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb"""
    import re
    
    tweet = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>", tweet)

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

class CustomTokenizer:
    def __init__(self, embedding, tokenizer): 
        self.embedding = embedding
        self.tokenizer = tokenizer
    
    def __call__(self, data): 
        input_ids = [self.embedding.get_token_id(self.tokenizer.tokenize(tweet)) for tweet in data['text']] 
        return {'input_ids' : input_ids}

class CustomNeuralNetwork(nn.Module): 
    def __init__(self, cfg, embedding): 
        super(CustomNeuralNetwork, self).__init__()
        self.cfg = cfg
        self.embedding = embedding.get_embedding()
        self.layers = [nn.Flatten()]

        self.layers.append(nn.Linear(self.cfg.embedding.dim * self.cfg.data.max_length, self.cfg.nn.hidden_units))
        self.layers.append(nn.ReLU())

        for _ in range(self.cfg.nn.hidden_layers):
            self.layers.append(nn.Linear(self.cfg.nn.hidden_units, self.cfg.nn.hidden_units))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Dropout(self.cfg.nn.dropout_prob))

        self.layers.append(nn.Linear(self.cfg.nn.hidden_units, 1))
        self.layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*self.layers)

    def forward(self, input_ids): 
        x = self.embedding(input_ids)
        return self.layers(x) 

class CustomTrainDataset(torch.utils.data.Dataset): 
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = item['input_ids']
        label = item['label']
        return input_ids, label
    
class CustomTestDataset(torch.utils.data.Dataset): 
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = item['input_ids']
        return input_ids

class NNClassifier: 
    def __init__(self, config, embedding): 
        self.cfg = config
        self.model = CustomNeuralNetwork(self.cfg, embedding)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.nn.lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_data): 
        self.model.train()
        self.model.to(self.device)

        dataset = CustomTrainDataset(train_data)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg.nn.batch_size, shuffle=True)

        for _ in tqdm(range(self.cfg.nn.epochs), desc="Training the NN..."): 
            avg_loss = 0
            for input_ids, labels in data_loader: 
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids)
                labels = labels.view(-1, 1)

                loss = self.criterion(outputs, labels)
                avg_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print(f"Training Loss: {avg_loss / len(data_loader)}")
    
    def test(self, test_data): 
        self.model.eval()
        self.model.to(self.device)

        dataset = CustomTestDataset(test_data)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg.nn.batch_size, shuffle=False)
        
        results = []
        with torch.no_grad(): 
            for input_ids in tqdm(data_loader, desc="Testing the NN..."): 
                input_ids = input_ids.to(self.device)

                outputs = self.model(input_ids)
                outputs = outputs.cpu().detach().numpy()   

                result = np.where(outputs >= THRESHOLD, 1, 0).squeeze()
                results.append(result)
        return np.concatenate(results, axis=0)
    
if __name__ == "__main__": 
    from utils import load_config, set_seed, save_outputs
    from data_loader import TwitterDataset
    from glove import GloveEmbedding
    from nltk import TweetTokenizer
    import time
    from sklearn.metrics import classification_report

    cfg = load_config()
    set_seed(cfg.general.seed)

    wordsegment.load()
    twitter = TwitterDataset(cfg, preprocessor)
    embedding = GloveEmbedding(cfg)
    tokenizer = TweetTokenizer()
    model = NNClassifier(cfg, embedding)

    tokenized_dataset = twitter.tokenize(CustomTokenizer(embedding, tokenizer))

    model.train(tokenized_dataset["train"])
    model_outputs_eval = model.test(tokenized_dataset["eval"])
    model_outputs_test = model.test(tokenized_dataset["test"])

    report = classification_report(tokenized_dataset["eval"]["label"], model_outputs_eval)
    print("Eval report:", report)
    
    save_outputs(np.where(model_outputs_test == 0, -1, 1), cfg.general.run_id)
