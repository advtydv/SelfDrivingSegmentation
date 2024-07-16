import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch import optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.functional as TF
from tqdm import tqdm
import torch.nn.functional as F

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64,128,256,512]):
        super(UNet, self).__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
            
        self.bottle = DoubleConv(features[-1], features[-1]*2)
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
            
        x = self.bottle(x)
        skips = skips[::-1]
        
        for i in range(0,len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skips[i//2]
            
            if(skip_connection.shape != x.shape):
                x = TF.resize(x, skip_connection.shape[2:])
            
            x = torch.cat((x, skip_connection), dim=1)
            x = self.ups[i+1](x)
            
        return self.final(x)
        
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.convs(x)
    
def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(f"shape of input is {x.shape}")
    print(f"shape of preds is {preds.shape}")
    
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()