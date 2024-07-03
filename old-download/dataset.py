import pandas as pd
import numpy as np
pos = pd.read_csv('twitter-datasets/train_pos.txt',sep='\t', header=None, names=['x'])
pos['label'] = np.ones(pos.shape[0], dtype=int)

neg = pd.read_csv('twitter-datasets/train_neg.txt',sep='\t', header=None, names=['x'])
neg['label'] = -np.ones(neg.shape[0], dtype=int)

df = pd.concat([pos, neg])
