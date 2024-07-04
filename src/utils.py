import os
import random

from box import Box
import numpy as np
import yaml
import torch


def load_config():
    """
    Loads the global configuration file.
    """
    with open('config.yaml', 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return Box(config)


def set_seed(seed):
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
