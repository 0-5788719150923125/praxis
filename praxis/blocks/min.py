import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.modules.dense import PraxisMLP
from praxis.modules.recurrent import minGRU


class PraxisGRU(nn.Module):
    """
    A minimally-recurrent network.
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig", *args, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size)
        self.recurrent = minGRU(config.hidden_size, expansion_factor=1.0, proj_out=None)
        self.ffn = PraxisMLP(config)

    def forward(self, x: Tensor, *args, **kwargs):
        out, _ = self.recurrent(self.norm(x))
        return self.ffn(stretch(out, target_min=-1.0)) + x, None, 0


def stretch(
    x: torch.Tensor, target_min: float = None, target_max: float = None
) -> torch.Tensor:
    """
    Stretches tensor values using linear interpolation.
    At least one target bound must be specified.
    """
    assert (
        target_min is not None or target_max is not None
    ), "At least one target bound must be specified"

    max_val = x.max()
    min_val = x.min()

    # If only target_min specified
    if target_max is None:
        # Keep max_val fixed, stretch everything else to target_min
        progress = (x - min_val) / (max_val - min_val)
        return target_min + progress * (max_val - target_min)

    # If only target_max specified
    if target_min is None:
        # Keep min_val fixed, stretch everything else to target_max
        progress = (x - min_val) / (max_val - min_val)
        return min_val + progress * (target_max - min_val)

    # If both targets specified
    progress = (x - min_val) / (max_val - min_val)
    return target_min + progress * (target_max - target_min)


if __name__ == "__main__":
    from dataclasses import dataclass

    import torch

    # Create a mock config
    @dataclass
    class MockConfig:
        hidden_size: int = 64
        activation: str = "gelu"
        dropout: float = 0

    # Initialize model and test input
    config = MockConfig()
    model = PraxisGRU(config)

    # Create sample input (batch_size=2, sequence_length=3, hidden_size=64)
    x = torch.randn(2, 3, 64)

    # Run forward pass
    output, hidden, loss = model(x)

    # Verify output
    print("=== Smoke Test Results ===")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden state: {hidden}")
    print(f"Loss value: {loss}")

    # Basic assertions
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    assert hidden is None, "Expected hidden state to be None"
    assert loss == 0, "Expected loss to be 0"

    print("All tests passed! 🎉")
