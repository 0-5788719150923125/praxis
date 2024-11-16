import torch.nn as nn

from praxis.modules.encoding import ALiBi, NoPE, RoPE, YaRN

ENCODING_REGISTRY = {"alibi": ALiBi, "nope": NoPE, "rope": RoPE, "yarn": YaRN}


class MultiIdentity(nn.Module):
    """
    Similar to nn.Identity(), except it will return the same number
    of outputs as inputs.
    """

    __version__ = "0.1.0"

    def forward(self, x, *args):
        return (x,) + args
