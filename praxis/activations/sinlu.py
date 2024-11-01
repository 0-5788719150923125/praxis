import torch
import torch.nn.functional as F
from torch import nn


class SinLU(nn.Module):
    """
    Implements SinLU, which has an interesting shape and learnable parameters:
    https://arxiv.org/abs/2306.01822
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1))
        self.beta = nn.Parameter(torch.empty(1))
        nn.init.constant_(self.alpha, 1.0)
        nn.init.constant_(self.beta, 1.0)

    def forward(self, x):
        return F.sigmoid(x) * (x + self.alpha * torch.sin(self.beta * x))
