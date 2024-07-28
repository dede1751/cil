import os
import random

from box import Box
import numpy as np
import yaml
import torch
import evaluate

THRESHOLD = 0.5

def load_config() -> Box:
    """
    Loads the global configuration file.
    """
    with open('config.yaml', 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return Box(config)


def set_seed(seed: int):
    """
    Sets random number generator seeds for PyTorch and NumPy to ensure reproducibility of results.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_outputs(outputs: np.ndarray, file_name: str):
    """
    Save the model outputs to a file.
    :param outputs: Model outputs, np.ndarry with shape (10'000,) and values in {-1, 1}
    """
    indices = np.arange(1, outputs.shape[0] + 1)
    combined = np.column_stack((indices, outputs))

    np.savetxt(
        f'{file_name}.csv', combined, delimiter=',', fmt='%d', header="Id,Prediction", comments='')

def compute_metrics(eval_pred):
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")
    predictions, labels = eval_pred

    predictions = (predictions >= THRESHOLD).astype(int)
    labels = labels.astype(int)

    return {
        "accuracy": load_accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": load_f1.compute(predictions=predictions, references=labels)["f1"]
    }