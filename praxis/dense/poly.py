import math
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor

from praxis.activations import ACT2CLS, ACT2FN

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class PolynomialExpansionMLP(nn.Module):
    """
    A novel dense layer based on explicit polynomial feature expansions.
    Learns combinations of features up to a specified degree while maintaining
    computational efficiency through careful parameter sharing. Polynomial
    functions are universal approximators (Stone-Weierstrass theorem).
    """

    __version__ = "0.1.0"

    def __init__(
        self,
        config: ConfigType,
        degree: int = 6,
        bottleneck: float = 0.5,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a polynomial expansion layer.

        Args:
            config: Configuration object with model parameters
            degree: Maximum polynomial degree to compute
            bottleneck: Bottleneck ratio for dimension reduction
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        dim = config.hidden_size
        self.dim: int = dim
        self.degree: int = degree
        self.reduced_dim: int = int(dim * bottleneck)

        # Reduce dimension for efficiency
        self.down: nn.Linear = nn.Linear(dim, self.reduced_dim)

        # Learnable coefficients for each degree
        self.degree_coeffs: nn.ParameterList = nn.ParameterList(
            [nn.Parameter(torch.randn(self.reduced_dim) * 0.02) for _ in range(degree)]
        )

        # Learnable mixing matrices for cross-terms
        self.cross_terms: nn.ParameterList = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.reduced_dim, self.reduced_dim) * 0.02)
                for _ in range(degree - 1)
            ]
        )

        # Project back to original dimension
        self.up: nn.Linear = nn.Linear(self.reduced_dim, dim)
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        Forward pass through the polynomial layer.

        Args:
            inputs: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after polynomial processing
        """
        residual = inputs

        # Project to lower dimension
        x = self.down(inputs)

        # Compute powers and cross-terms
        terms: List[Tensor] = []
        x_power = x

        for i in range(self.degree):
            # Add direct power term
            weighted_power = x_power * self.degree_coeffs[i]
            terms.append(weighted_power)

            # Add cross-terms for higher degrees
            if i < self.degree - 1:
                cross_term = torch.matmul(x_power, self.cross_terms[i]) * x
                terms.append(cross_term)

            # Prepare next power
            x_power = x_power * x

        # Combine all terms
        output = sum(terms)

        # Project back to original dimension
        output = self.up(output)
        output = self.dropout(output)

        # Residual connection
        return output + residual
