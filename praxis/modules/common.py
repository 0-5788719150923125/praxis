import torch.nn as nn


class MultiIdentity(nn.Module):
    """
    Similar to nn.Identity(), except it will return the same number
    of outputs as inputs.
    """

    __version__ = "0.1.0"

    def forward(self, x, *args):
        return (x,) + args
