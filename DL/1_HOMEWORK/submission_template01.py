import numpy as np
import torch
from torch import nn


def create_model():
    NN = nn.Sequential(nn.Linear(784, 256, bias=True),
                       nn.ReLU(),
                       nn.Linear(256, 16, bias=True),
                       nn.ReLU(),
                       nn.Linear(16, 10, bias=True),
                       )
    return NN


model = create_model()


def count_parameters(model):
    return sum([param.numel() for param in model.parameters()])

