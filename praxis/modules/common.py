import torch.nn as nn


class MultiIdentity(nn.Module):
    def forward(self, x, *args):
        return (x,) + args
