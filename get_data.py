#%%
from keras.datasets import mnist 
from random import shuffle, seed
import itertools
from_iterable = itertools.chain.from_iterable

import torch
from torch import nn
from torch.optim import Adam
from torchinfo import summary as torch_summary



(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = torch.from_numpy(train_x); train_y = torch.from_numpy(train_y)
test_x  = torch.from_numpy(test_x);  test_y  = torch.from_numpy(test_y)

xs = torch.cat([train_x, test_x]).unsqueeze(-1) /255
ys = torch.cat([train_y, test_y])



seed(100)
indexes = [i for i in range(len(xs))]
shuffle(indexes)

test_indexes = []
for i in range(20):
    test_indexes.append(indexes[3500*i : 3500*(i+1)])
train_indexes = []
for k in range(20):
    train_indexes.append(list(from_iterable(test_indexes[k_] for k_ in range(20) if k_ != k)))



def get_batch(k, batch_size = 128, test = False):
    if(test):
        x = xs[test_indexes[k]]
        y = ys[test_indexes[k]]
        return(x, y)
    indexes = train_indexes[k]
    shuffle(indexes)
    batch = indexes[:batch_size]
    x = xs[batch]
    y = ys[batch]
    return(x, y)
# %%
