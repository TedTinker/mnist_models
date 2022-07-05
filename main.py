#%%
from utils import models, plot_boxes
from train_test import train_test

from models.model_a1 import a1_list
from models.model_a2 import a2_list
from models.model_a3 import a3_list
from models.model_a4 import a4_list

from models.model_b1 import b1_list
from models.model_b2 import b2_list
from models.model_b3 import b3_list
from models.model_b4 import b4_list

from models.model_c1 import c1_list



test_losses = {m : [] for m in models}

for model in a1_list: test_losses["a1"].append(train_test(model))
for model in a2_list: test_losses["a2"].append(train_test(model))
for model in a3_list: test_losses["a3"].append(train_test(model))
for model in a4_list: test_losses["a4"].append(train_test(model))

for model in b1_list: test_losses["b1"].append(train_test(model))
for model in b2_list: test_losses["b2"].append(train_test(model))
for model in b3_list: test_losses["b3"].append(train_test(model))
for model in b4_list: test_losses["b4"].append(train_test(model))

for model in c1_list: test_losses["c1"].append(train_test(model))

plot_boxes(test_losses)
# %%
