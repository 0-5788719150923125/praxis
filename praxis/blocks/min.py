import math
from typing import Any, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.dense import DENSE_REGISTRY
from praxis.recurrent import RECURRENT_REGISTRY

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class MinGRUBlock(nn.Module):
    """A block built around min-GRU (Were RNNs All We Needed?,
    arxiv 2410.01201): a stripped-down GRU with no hidden-state dependence
    in the gates, which makes it amenable to parallel scan. Pairs the min-GRU
    with a standard MLP and pre-norm residual.
    """

    def __init__(self, config: ConfigType, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size)
        self.recurrent = RECURRENT_REGISTRY["min_gru"](
            config.hidden_size, expansion_factor=1.0, proj_out=None
        )
        self.ffn = DENSE_REGISTRY.get("mlp")(config)

    def forward(
        self,
        inputs: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], float]:
        """
        Forward pass through the MinGRUBlock block.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Tuple containing:
                - Output tensor
                - None (no past key values)
                - None (no layer state)
                - Zero loss value
        """
        out, _ = self.recurrent(self.norm(inputs))
        return self.ffn(stretch(out, target_min=-1.0)) + inputs, None, None, 0


def stretch(
    x: Tensor, target_min: Optional[float] = None, target_max: Optional[float] = None
) -> Tensor:
    """
    Stretches tensor values using linear interpolation.
    At least one target bound must be specified.

    The range is each position's own, over the feature axis: a range taken over
    the whole tensor would let a later token, or another row, rescale this one.
    """
    assert (
        target_min is not None or target_max is not None
    ), "At least one target bound must be specified"

    max_val = x.amax(dim=-1, keepdim=True)
    min_val = x.amin(dim=-1, keepdim=True)
    span = (max_val - min_val).clamp_min(torch.finfo(x.dtype).eps)

    progress = (x - min_val) / span

    # If only target_min specified
    if target_max is None:
        # Keep max_val fixed, stretch everything else to target_min
        return target_min + progress * (max_val - target_min)

    # If only target_max specified
    if target_min is None:
        # Keep min_val fixed, stretch everything else to target_max
        return min_val + progress * (target_max - min_val)

    # If both targets specified
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
    model = MinGRUBlock(config)

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
