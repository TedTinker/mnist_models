#%%
import time
import torch
import torch.nn.functional as F
from torch.nn import NLLLoss

from utils import device, plot_loss_acc, save_model, k, epochs
from get_data import get_batch

def train_test(model, M, K, E, batch_size = 128, show_after = 100):
    train_losses = []; test_losses = []
    train_acc = [];    test_acc = []
    
    for e in range(1,epochs+1):
        E.update()
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
                plot_loss_acc(model, e, train_losses, test_losses, train_acc, test_acc)
                if(E.count == epochs):
                    E.count = 0; E.start = time.time()
                    K.update()
                    if(K.count == k):
                        K.count = 0; K.start = time.time()
                        M.update()
                #save_model(model, e)

                
    return(train_losses[-1], test_losses[-1])
# %%
