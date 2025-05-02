import math
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor

from praxis.activations import ACT2CLS, ACT2FN

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class GatedLinearMLP(nn.Module):
    """
    A standard MLP, augmented with Gated Linear Units.
    """

    __version__ = "0.1.0"

    def __init__(
        self,
        config: ConfigType,
        activation: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a GLU-based MLP module.

        Args:
            config: Configuration object with model parameters
            activation: Activation function name (from ACT2CLS registry)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        activation = activation or config.activation

        # First calculate the target size after chunking (down projection input size)
        down_size = int((4 / 3) * config.hidden_size)
        # Double it for up projection to ensure chunks match
        up_size = 2 * down_size

        self.up: nn.Linear = nn.Linear(config.hidden_size, up_size)
        self.act: nn.Module = ACT2CLS[activation](*args, **kwargs)
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)
        self.down: nn.Linear = nn.Linear(down_size, config.hidden_size)

    def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        Forward pass through the GLU module.

        Args:
            inputs: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after GLU processing
        """
        a, b = self.up(inputs).chunk(2, dim=-1)
        return self.down(self.dropout(a * self.act(b)))
