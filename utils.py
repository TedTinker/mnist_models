#%%
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(device.type == "cpu"):   print("\n\nAAAAAAAUGH! CPU! >:C\n")
else:                       print("\n\nUsing CUDA! :D\n")

def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass
    
    
    
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 

if not os.path.exists('plots'):
    os.makedirs('plots')

folders = ["loss", "accuracy", "model"]
models = ["a1", "a2", "a3", "a4",
          "b1", "b2", "b3", "b4", 
          "c1", "c2", "c3", "c4"]

for folder in folders:
    if not os.path.exists('plots/{}'.format(folder)):
        os.makedirs('plots/{}'.format(folder))
        for m in models:
            if not os.path.exists('plots/{}/{}'.format(folder, m)):
                os.makedirs('plots/{}/{}'.format(folder, m))

for folder in folders:
    for m in models:
        for f in os.listdir("plots/{}/{}".format(folder, m)):
            os.remove("plots/{}/{}/{}".format(folder, m, f))
    
    

import matplotlib.pyplot as plt

def plot_losses(model, e, train_losses, test_losses):
    plt.plot(train_losses, color = "blue", label='Train')
    plt.plot(test_losses,  color = "red",  label='Test')
    plt.title("{}: {} epochs loss".format(model.k, e))
    plt.legend()
    
    plt.savefig("plots/loss/{}/{}_{}".format(model.name[:2], model.name, e))
    plt.show()
    plt.close()
    
def plot_accuracy(model, e, train_acc, test_acc):
    plt.plot(train_acc, color = "blue", label='Train')
    plt.plot(test_acc,  color = "red",  label='Test')
    plt.title("{}: {} epochs accuracy".format(model.k, e))
    plt.ylim((0,100))
    plt.legend()
    
    plt.savefig("plots/accuracy/{}/{}_{}".format(model.name[:2], model.name, e))
    plt.show()
    plt.close()
    
def plot_boxes(test_losses):
    pass
    
def save_model(model, e):
    torch.save(model, "plots/model/{}/{}_{}".format(model.name[:2], model.name, e))
# %%
