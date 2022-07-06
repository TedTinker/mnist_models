#%%
import torch
from torch import nn 
from torch.optim import Adam
from torchinfo import summary as torch_summary

utils = True
try: from utils import device, init_weights, k
except: utils = False; k = 20

class B3(nn.Module):
    
    def __init__(self, k):
        super().__init__()
        
        self.name = "b3_{}".format(str(k+1).zfill(3))
        self.k = k
                
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels = 1, 
                out_channels = 16, 
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(
                in_channels = 16, 
                out_channels = 16, 
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size = 2))
        
        example = torch.zeros((1, 1, 28, 28))
        example = self.cnn(example).flatten(1)
        quantity = example.shape[-1]
        
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
        return(y)

b3_list = []
for k_ in range(k):
    b3_list.append(B3(k_))
    
if __name__ == "__main__":
    print(b3_list[0])
    print()
    print(torch_summary(b3_list[0], (10, 28, 28, 1)))
# %%
