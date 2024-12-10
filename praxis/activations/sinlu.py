import torch
import torch.nn.functional as F
from torch import nn


class SinLU(nn.Module):
    """
    Implements SinLU, which has an interesting shape and learnable parameters:
    https://www.mdpi.com/2227-7390/10/3/337
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.sigmoid(x) * (x + self.alpha * torch.sin(self.beta * x))
