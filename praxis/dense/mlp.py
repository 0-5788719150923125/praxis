import math
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor

from praxis.activations import ACT2CLS, ACT2FN

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class MultiLayerPerceptron(nn.Sequential):
    """
    A standard Multi-Layer Perceptron.
    """

    __version__ = "0.1.0"

    def __init__(
        self,
        config: ConfigType,
        activation: Optional[str] = None,
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a standard MLP module.

        Args:
            config: Configuration object with model parameters
            activation: Activation function name (from ACT2FN registry)
            input_dim: Input dimension size
            hidden_dim: Hidden dimension size
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        activation = activation or config.activation
        input_dim = input_dim or config.hidden_size
        hidden_dim = hidden_dim or input_dim * 4
        super().__init__(
            OrderedDict(
                [
                    ("up", nn.Linear(input_dim, hidden_dim)),
                    ("act", ACT2FN[activation]),
                    ("dropout", nn.Dropout(config.dropout)),
                    ("down", nn.Linear(hidden_dim, input_dim)),
                ]
            )
        )

    def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        Forward pass through the MLP.

        Args:
            inputs: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after MLP processing
        """
        return super().forward(inputs)
