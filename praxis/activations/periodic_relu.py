import torch
from torch import nn


class PeriodicReLU(nn.Module):
    def __init__(self, period=2 * torch.pi):
        super().__init__()
        self.period = period

    def forward(self, x):
        # Calculate phase within the period
        phase = x - self.period * torch.floor(x / self.period + 0.5)

        # First triangle wave (shifted by quarter period)
        shifted = phase + self.period / 4
        shifted_phase = shifted - self.period * torch.floor(shifted / self.period + 0.5)
        tri1 = shifted_phase * torch.pow(-1.0, torch.floor(shifted / self.period + 0.5))

        # Second triangle wave
        tri2 = phase * torch.pow(-1.0, torch.floor(x / self.period + 0.5))

        # Combine with proper scaling according to the paper
        return (torch.pi / 4) * (tri1 + tri2)
