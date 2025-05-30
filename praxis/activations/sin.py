import torch
from torch import Tensor, nn

torch.sqrt2 = 1.414213562  # sqrt 2 \approx 1.414213562


class Sine(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sqrt2 * torch.sin(x)
