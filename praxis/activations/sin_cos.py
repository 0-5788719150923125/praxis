import torch
from torch import nn


class SineCosine(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        # The paper mentioned this simpler version can work well
        # and has potential advantages over just sine activation
        return self.scale * (torch.sin(x) + torch.cos(x))
