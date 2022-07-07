#%%
from keras.datasets import mnist 
from random import shuffle, seed
import itertools
from_iterable = itertools.chain.from_iterable

import torch

from utils import k



(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = torch.from_numpy(train_x); train_y = torch.from_numpy(train_y)
test_x  = torch.from_numpy(test_x);  test_y  = torch.from_numpy(test_y)

xs = torch.cat([train_x, test_x]).unsqueeze(-1)/255
ys = torch.cat([train_y, test_y])

data_len = len(xs)


seed(100)
indexes = [i for i in range(len(xs))]
shuffle(indexes)

test_indexes = []
for i in range(k):
    test_indexes.append(indexes[i*data_len//k : (i+1)*data_len//k])
train_indexes = []
for k_ in range(k):
    train_indexes.append(list(from_iterable(test_indexes[k__] for k__ in range(k) if k__ != k_)))



def get_batch(k_, batch_size = 128, test = False):
    if(test):
        x = xs[test_indexes[k_]]
        y = ys[test_indexes[k_]]
        return(x, y)
    indexes = train_indexes[k_]
    shuffle(indexes)
    batch = indexes[:batch_size]
    x = xs[batch]
    y = ys[batch]
    return(x, y)

if __name__ == "__main__":
    x, y = get_batch(0)
    print(x.shape, y.shape)
    x, y = get_batch(0, test = True)
    print(x.shape, y.shape)
# %%
