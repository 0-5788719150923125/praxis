import torch
from torch import nn


class JaggedSine(nn.Module):
    def __init__(self, frequencies=[1.0, 3.0, 5.0], amplitudes=[1.0, 0.3, 0.1]):
        super().__init__()
        assert len(frequencies) == len(
            amplitudes
        ), "Must have same number of frequencies and amplitudes"
        self.register_buffer("frequencies", torch.tensor(frequencies))
        self.register_buffer("amplitudes", torch.tensor(amplitudes))

    def forward(self, x):
        # Add a new dimension to x for the frequencies
        # If x is [batch_size, features], it becomes [batch_size, features, 1]
        # If x is just [features], it becomes [features, 1]
        x_expanded = x.unsqueeze(-1)

        # Add a new dimension to frequencies/amplitudes for broadcasting
        # [freq_count] becomes [1, freq_count]
        freqs = self.frequencies.view(1, -1)
        amps = self.amplitudes.view(1, -1)

        # Now broadcasting will work correctly
        return (amps * torch.sin(freqs * x_expanded)).sum(dim=-1)
