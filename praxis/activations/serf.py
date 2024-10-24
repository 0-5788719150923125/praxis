import torch
import torch.nn.functional as F
from torch import nn


class SERF(nn.Module):
    """
    Implements the SERF activation function, as described in:
    https://arxiv.org/abs/2108.09598
    """

    def forward(self, x):
        return x * torch.erf(torch.log(1 + torch.exp(x)))
