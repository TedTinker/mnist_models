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



train_losses = {m : [] for m in models}
test_losses  = {m : [] for m in models}

for model in a1_list: 
    train_loss, test_loss = train_test(model)
    train_losses["a1"].append(train_loss)
    test_losses["a1"].append(test_loss)
for model in a2_list: 
    train_loss, test_loss = train_test(model)
    train_losses["a2"].append(train_loss)
    test_losses["a2"].append(test_loss)
for model in a3_list: 
    train_loss, test_loss = train_test(model)
    train_losses["a3"].append(train_loss)
    test_losses["a3"].append(test_loss)
for model in a4_list: 
    train_loss, test_loss = train_test(model)
    train_losses["a4"].append(train_loss)
    test_losses["a4"].append(test_loss)
    
for model in b1_list: 
    train_loss, test_loss = train_test(model)
    train_losses["b1"].append(train_loss)
    test_losses["b1"].append(test_loss)
for model in b2_list: 
    train_loss, test_loss = train_test(model)
    train_losses["b2"].append(train_loss)
    test_losses["b2"].append(test_loss)
for model in b3_list: 
    train_loss, test_loss = train_test(model)
    train_losses["b3"].append(train_loss)
    test_losses["b3"].append(test_loss)
for model in b4_list: 
    train_loss, test_loss = train_test(model)
    train_losses["b4"].append(train_loss)
    test_losses["b4"].append(test_loss)

for model in c1_list: 
    train_loss, test_loss = train_test(model)
    train_losses["c1"].append(train_loss)
    test_losses["c1"].append(test_loss)
    
plot_boxes(test_losses)
# %%
