#%%
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import NLLLoss

from utils import device, plot_losses, plot_accuracy, save_model
from get_data import get_batch

def train_test(model, epochs = 100, batch_size = 128, show_after = 100):
    train_losses = []; test_losses = []
    train_acc = [];    test_acc = []
    
    for e in tqdm(range(1,epochs+1)):
        x, y = get_batch(k_ = model.k, batch_size = batch_size, test = False)
        predicted = model(x)
        loss = F.nll_loss(predicted, y.to(device))
        model.opt.zero_grad()
        loss.backward()
        model.opt.step()
        train_losses.append(loss.item())
        accurate = [True if torch.argmax(p).item() == y[i].item() else False for i, p in enumerate(predicted)]
        train_acc.append(100*sum(accurate)/len(accurate))
        
        with torch.no_grad():
            x, y = get_batch(k_ = model.k, test = True)
            predicted = model(x)
            loss = F.nll_loss(predicted, y.to(device))
            test_losses.append(loss.item())
            accurate = [True if torch.argmax(p).item() == y[i].item() else False for i, p in enumerate(predicted)]
            test_acc.append(100*sum(accurate)/len(accurate))

            if(e%show_after == 0 or e==epochs):
                plot_losses(model, e, train_losses, test_losses)
                plot_accuracy(model, e, train_acc, test_acc)
                #save_model(model, e)
                print()
                
    return(test_losses[-1])
# %%
