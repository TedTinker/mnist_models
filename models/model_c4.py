#%%
import torch
from torch import nn 
import torchgan.layers as gnn
from torch.optim import Adam
from torchinfo import summary as torch_summary

utils = True
try: from utils import device, init_weights, k
except: utils = False; k = 20

class C4(nn.Module):
    
    def __init__(self, k):
        super().__init__()
        
        self.name = "c4_{}".format(str(k+1).zfill(3))
        self.k = k
                
        self.cnn = nn.Sequential(
            gnn.ResidualBlock2d(
                filters = [1, 4, 8], 
                kernels = [3, 3], 
                paddings = [1, 1], 
                nonlinearity = nn.LeakyReLU(), 
                last_nonlinearity = nn.LeakyReLU()),
            nn.MaxPool2d(kernel_size = 2),
            gnn.ResidualBlock2d(
                filters = [8, 8, 8], 
                kernels = [3, 3], 
                paddings = [1, 1], 
                nonlinearity = nn.LeakyReLU(), 
                last_nonlinearity = nn.LeakyReLU()),
            nn.MaxPool2d(kernel_size = 2),
            gnn.SelfAttention2d(input_dims = 8),
            nn.LeakyReLU())
        
        example = torch.zeros((1, 1, 28, 28))
        example = self.cnn(example)
        quantity = example.flatten(1).shape[-1]
        
        self.lin = nn.Sequential(
            nn.Linear(
                in_features = quantity,
                out_features = 32),
            nn.LeakyReLU(),
            nn.Linear(
                in_features = 32,
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
        y = (y+1)/2
        return(y)

c4_list = []
for k_ in range(k):
    c4_list.append(C4(k_))
    
if __name__ == "__main__":
    print(c4_list[0])
    print()
    print(torch_summary(c4_list[0], (10, 28, 28, 1)))
# %%
