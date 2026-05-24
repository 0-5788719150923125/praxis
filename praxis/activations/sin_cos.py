import torch
from torch import Tensor, nn


class SineCosine(nn.Module):
    """``scale * (sin(x) + cos(x))`` - a phase-shifted sinusoid that gives
    the network a non-zero gradient at the origin, unlike pure ``sin``."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        # The paper mentioned this simpler version can work well
        # and has potential advantages over just sine activation
        return self.scale * (torch.sin(x) + torch.cos(x))
