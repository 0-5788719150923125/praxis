import torch
from torch import Tensor, nn


class SineCosine(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        # The paper mentioned this simpler version can work well
        # and has potential advantages over just sine activation
        return self.scale * (torch.sin(x) + torch.cos(x))
