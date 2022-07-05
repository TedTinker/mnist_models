#%%
from utils import models, plot_boxes
from train_test import train_test
from model_a1 import a1_list

test_losses = {m : [] for m in models}
for model in a1_list:
    test_losses["a1"].append(train_test(model))

plot_boxes(test_losses)
# %%
