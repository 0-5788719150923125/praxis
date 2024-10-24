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
        self.a = nn.Parameter(torch.empty(1))
        self.b = nn.Parameter(torch.empty(1))

        nn.init.constant_(self.a, 1.0)
        nn.init.constant_(self.b, 1.0)

    def forward(self, x):
        return F.sigmoid(x) * (x + self.a * torch.sin(self.b * x))
