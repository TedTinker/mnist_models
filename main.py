#%%
from tqdm import tqdm
import enlighten
from utils import models, plot_boxes, k, epochs
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



train_losses = {m : [] for m in models}
test_losses  = {m : [] for m in models}

models.sort()
model_lists = [a1_list, a2_list, a3_list, a4_list,
               b1_list, b2_list, b3_list, b4_list,
               c1_list]

manager = enlighten.get_manager()
M = manager.counter(total = len(models), desc = "Models:", unit = "ticks", color = "red")
K = manager.counter(total = k,           desc = "K:",      unit = "ticks", color = "blue")
E = manager.counter(total = epochs,      desc = "Epochs:", unit = "ticks", color = "green")

for list, name in zip(model_lists, models):
    for model in list:
        train_loss, test_loss = train_test(model, M, K, E)
        train_losses[name].append(train_loss)
        test_losses[name].append(test_loss)
    
plot_boxes(train_losses, test_losses)
# %%
