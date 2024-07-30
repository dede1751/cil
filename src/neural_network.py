import torch
import torch.nn as nn
import numpy as np 
import utils

from tqdm import tqdm

class NeuralNetwork(nn.Module):
    def __init__(self, device): 
        super(NeuralNetwork, self).__init__()
        self.device = device

    def run_test(self, test_dl):
        # Evaluate the model on the val set
        self.eval()  # Set the model to evaluation mode
        predictions = np.array([])
        with torch.no_grad():  # Disable gradient calculation for evaluation
            with tqdm(enumerate(test_dl), total=len(test_dl), desc=f"Testing") as pbar:
                for batch, x in pbar:
                    x.to(self.device)
                    y_pred = self(x.input_ids).view(-1)
                    predicted = (y_pred > utils.THRESHOLD).float() 
                    predicted = predicted * 2 - 1

                    predictions = np.concatenate((predictions, predicted.cpu()))

        return predictions


    def run_eval(self, val_dl, epoch, num_epochs):
        # Evaluate the model on the val set
        self.eval()  # Set the model to evaluation mode
        predictions = np.array([])
        labels = np.array([])
        with torch.no_grad():  # Disable gradient calculation for evaluation
            with tqdm(enumerate(val_dl), total=len(val_dl), desc=f"Validation {epoch+1}/{num_epochs}") as pbar:
                for batch, x in pbar:
                    x.to(self.device)
                    y_pred = self(x.input_ids).view(-1)
                    
                    predictions = np.concatenate((predictions, y_pred.cpu()))
                    labels = np.concatenate((labels, x.labels.cpu()))

        # Calculate accuracy
        metrics = utils.compute_metrics((predictions, labels))
        print(f"Epoch {epoch}/{num_epochs}, Val Accuracy: {metrics['accuracy']:.4f}, Val F1: {metrics['f1']:.4f}")
        return metrics

    def run_train(self, train_dl, optimizer, criterion, epoch, num_epochs):
        # Initialize the progress bar for the current epoch
        self.train()
        with tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch}/{num_epochs}") as pbar:
            for batch, x in pbar:
                x.to(self.device)
                y_pred = self(x.input_ids).view(-1)
                loss = criterion(y_pred, x.labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': loss.item()})