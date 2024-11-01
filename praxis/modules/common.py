import torch.nn as nn


class MultiIdentity(nn.Module):

    __version__ = "0.1.0"

    def forward(self, x, *args):
        return (x,) + args
