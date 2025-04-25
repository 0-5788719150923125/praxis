import torch
from torch import nn, Tensor

torch.pi = 3.1415926535897932
torch.pdiv2 = 1.570796326  # π/2
torch.pdiv4 = 0.785398163  # π/4


class PeriodicReLU(nn.Module):
    """
    Stolen from here:
    https://github.com/AaltoML/PeriodicBNN/blob/main/python_codes/model.py
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.pdiv4 * (
            self._triangle_activation(x) + self._triangle_activation(x + torch.pdiv2)
        )

    def _triangle_activation(self, x: Tensor) -> Tensor:
        return (x - torch.pi * torch.floor(x / torch.pi + 0.5)) * (-1) ** torch.floor(
            x / torch.pi + 0.5
        )
