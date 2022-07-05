#%%
from torch import nn 
from torch.optim import Adam
from torchinfo import summary as torch_summary

utils = True
try: from utils import device, init_weights, k
except: utils = False; k = 20

class A3(nn.Module):
    
    def __init__(self, k):
        super().__init__()
        
        self.name = "a3_{}".format(str(k+1).zfill(3))
        self.k = k
        
        self.lin = nn.Sequential(
            nn.Linear(
                in_features = 28*28,
                out_features = 128),
            nn.LeakyReLU(),
            nn.Linear(
                in_features = 128,
                out_features = 10),
            nn.LogSoftmax(1))
        
        if(utils):
            init_weights(self.lin)
            self.to(device)
        self.opt = Adam(self.parameters())
        
    def forward(self, x):
        if(utils): x = x.to(device)
        x = (x*2) - 1
        x = x.flatten(1)
        y = self.lin(x)
        y = (y+1)/2
        return(y)

a3_list = []
for k_ in range(k):
    a3_list.append(A3(k_))
    
if __name__ == "__main__":
    print(a3_list[0])
    print()
    print(torch_summary(a3_list[0], (10, 28, 28)))
# %%