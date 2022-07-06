#%%

k = 20
epochs = 100

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

models = os.listdir("models")
models = [m[6:-3] for m in models if m != "__pycache__"]

for m in models:
    if not os.path.exists('plots/{}'.format(m)):
        os.makedirs('plots/{}'.format(m))

for m in models:
    for file in os.listdir("plots/{}".format(m)):
        os.remove("plots/{}/{}".format(m, file))
    
    

import matplotlib.pyplot as plt

def plot_loss_acc(model, e, train_losses, test_losses, train_acc, test_acc):
    fig, ax1 = plt.subplots()
    fig.set_figsize=(7,7)
    ax2 = ax1.twinx()
    ax1.plot(train_losses, color = "b", label = 'Train loss')
    ax1.plot(test_losses,  color = "r", label = 'Test loss')
    ax1.set_ylabel("Loss")
    ax2.plot(train_acc, color = "c", label = 'Train acc')
    ax2.plot(test_acc,  color = "m", label = 'Test acc')
    ax2.set_ylabel("Accuracy")
    plt.title("{}: {} epochs loss and accuracy".format(model.name, e))
    ax2.set_ylim((0,100))
    ax1.legend(loc='upper left')
    ax2.legend(loc='lower left')
    
    plt.savefig("plots/{}/{}_{}".format(model.name[:2], model.name, e))
    #plt.show()
    plt.close()
    
def plot_boxes_loss(train_losses, test_losses):
    plt.figure(figsize=(15,7))
    train_c = (0,0,1,.1)
    k_train = list(train_losses.keys())
    v_train = list(train_losses.values())
    k_train, v_train = zip(*sorted(zip(k_train, v_train)))
    train = plt.boxplot(v_train, vert = True, widths = .75,
        patch_artist=True,
        boxprops=dict(facecolor=train_c, color=train_c),
        capprops=dict(color=train_c),
        whiskerprops=dict(color=train_c),
        flierprops=dict(color=train_c, markeredgecolor=train_c),
        medianprops=dict(color=train_c))
    
    test_c = (1,0,0,.5)
    k_test = list(test_losses.keys())
    v_test = list(test_losses.values())
    k_test, v_test = zip(*sorted(zip(k_test, v_test)))
    test = plt.boxplot(v_test, vert = True, widths = .25,
        patch_artist=True,
        boxprops=dict(facecolor=test_c, color=test_c),
        capprops=dict(color=test_c),
        whiskerprops=dict(color=test_c),
        flierprops=dict(color=test_c, markeredgecolor=test_c),
        medianprops=dict(color=test_c))
    
    plt.xticks(ticks = [i for i in range(1, len(k_test)+1)], labels = k_test)
    plt.title("Model losses")
    
    between_letters = []
    ongoing_letter = ""
    for i, name in enumerate(k_test):
        if(name[0] != ongoing_letter):
            between_letters.append(i+.5)
            ongoing_letter = name[0]
    for x in between_letters:
        plt.axvline(x=x, color = "black", linewidth = 1, linestyle = "-")
    plt.legend([train["boxes"][0], test["boxes"][0]], ['Train losses', 'Test losses'], loc='lower left')

    plt.savefig("plots/boxes_loss")
    plt.show()
    plt.close()
    
def plot_boxes_acc(train_acc, test_acc):
    plt.figure(figsize=(15,7))
    train_c = (0,0,1,.1)
    k_train = list(train_acc.keys())
    v_train = list(train_acc.values())
    k_train, v_train = zip(*sorted(zip(k_train, v_train)))
    train = plt.boxplot(v_train, vert = True, widths = .75,
        patch_artist=True,
        boxprops=dict(facecolor=train_c, color=train_c),
        capprops=dict(color=train_c),
        whiskerprops=dict(color=train_c),
        flierprops=dict(color=train_c, markeredgecolor=train_c),
        medianprops=dict(color=train_c))
    
    test_c = (1,0,0,.5)
    k_test = list(test_acc.keys())
    v_test = list(test_acc.values())
    k_test, v_test = zip(*sorted(zip(k_test, v_test)))
    test = plt.boxplot(v_test, vert = True, widths = .25,
        patch_artist=True,
        boxprops=dict(facecolor=test_c, color=test_c),
        capprops=dict(color=test_c),
        whiskerprops=dict(color=test_c),
        flierprops=dict(color=test_c, markeredgecolor=test_c),
        medianprops=dict(color=test_c))
    
    plt.xticks(ticks = [i for i in range(1, len(k_test)+1)], labels = k_test)
    plt.title("Model accuracies")
    
    between_letters = []
    ongoing_letter = ""
    for i, name in enumerate(k_test):
        if(name[0] != ongoing_letter):
            between_letters.append(i+.5)
            ongoing_letter = name[0]
    for x in between_letters:
        plt.axvline(x=x, color = "black", linewidth = 1, linestyle = "-")
    plt.legend([train["boxes"][0], test["boxes"][0]], ['Train accuracies', 'Test accuracies'], loc='lower left')
    
    minimums = [min(l) for l in list(train_acc.values()) + list(test_acc.values())]
    minimum = min(minimums)
    plt.ylim((minimum-1,101))

    plt.savefig("plots/boxes_acc")
    plt.show()
    plt.close()
# %%
