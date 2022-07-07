#%%
import torch
from torch import nn 
import torchgan.layers as gnn
from torch.optim import Adam
from torchinfo import summary as torch_summary

utils = True
try: from utils import device, init_weights, k, delete_these
except: 
    utils = False; k = 20
    def delete_these(verbose, *args): pass

class C1(nn.Module):
    
    def __init__(self, k):
        super().__init__()
        
        self.name = "c1_{}".format(str(k+1).zfill(3))
        self.k = k
        
        depth = 2
        self.cnn = nn.Sequential(
            gnn.DenseBlock2d(
                depth = depth,
                in_channels = 1,
                growth_rate = 16,
                block = gnn.BasicBlock2d,
                kernel = 3,
                padding = 1,
                batchnorm = False),
            gnn.TransitionBlock2d(
                in_channels = 1 + 16*depth,
                out_channels = 16,
                kernel = 1,
                batchnorm = False),
            nn.MaxPool2d(kernel_size = 2))
        
        example = torch.zeros((1, 1, 28, 28))
        example = self.cnn(example)
        quantity = example.flatten(1).shape[-1]
        
        self.lin = nn.Sequential(
            nn.Linear(
                in_features = quantity,
                out_features = 10),
            nn.LogSoftmax(1))
        
        if(utils):
            init_weights(self.lin)
            self.to(device)
        self.opt = Adam(self.parameters())
        
    def forward(self, x):
        if(utils): x = x.to(device)
        x = (x*2) - 1
        x = x.permute(0, -1, 1, 2)
        x = self.cnn(x).flatten(1)
        y = self.lin(x)
        delete_these(False, x)
        return(y.cpu())

c1_list = []
for k_ in range(k):
    c1_list.append(C1(k_))
    
if __name__ == "__main__":
    print(c1_list[0])
    print()
    print(torch_summary(c1_list[0], (10, 28, 28, 1)))
# %%
