import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # размер исходной картинки 32х32

        # conv 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3,3)) #30x30
        # pool
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2)) #15x15
        # conv 2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=(3,3)) #13x13
        
        # flatten
        self.flatten = nn.Flatten()

        # linear 1
        self.fc1 = nn.Linear(13 * 13 * 9, 128)
        # linear 2
        self.fc2 = nn.Linear(128, 10)

    
    def forward(self, x):
        # forward pass сети

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
def create_model():
    return ConvNet()
