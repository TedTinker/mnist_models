#%%
import torch
from torch import nn 
from torch.optim import Adam
from torchinfo import summary as torch_summary

from utils import device, init_weights

class A1(nn.Module):
    
    def __init__(self, k):
        super().__init__()
        
        self.name = "a1_{}".format(str(k).zfill(3))
        self.k = k
        
        self.lin = nn.Sequential(
            nn.Linear(
                in_features = 28*28,
                out_features = 32),
            nn.LeakyReLU(),
            nn.Linear(
                in_features = 32,
                out_features = 10),
            nn.LogSoftmax(1))
        
        init_weights(self.lin)
        self.to(device)
        self.opt = Adam(self.parameters())
        
    def forward(self, x):
        x = x.to(device)
        x = (x*2) - 1
        x = x.flatten(1)
        y = self.lin(x)
        y = (y+1)/2
        return(y)

a1_list = []
for k in range(20):
    a1_list.append(A1(k))
    
if __name__ == "__main__":
    print(a1_list[0])
    print()
    print(torch_summary(a1_list[0], (10, 28, 28)))
# %%
