import torch
from torch import nn


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)
