import math
from collections import OrderedDict
from typing import Optional, OrderedDict, Tuple

import torch
import torch.nn as nn
from torch import nn

from praxis.activations import ACT2CLS, ACT2FN


class PraxisMLP(nn.Sequential):
    """
    A standard Multi-Layer Perceptron.
    """

    __version__ = "0.1.0"

    def __init__(
        self,
        config: "AutoConfig",
        activation=None,
        input_dim=None,
        hidden_dim=None,
        *args,
        **kwargs,
    ):
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

    def forward(self, inputs, *args, **kwargs):
        return super().forward(inputs)


class PraxisGLU(nn.Module):
    """
    A standard MLP, augmented with Gated Linear Units.
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig", activation=None, *args, **kwargs):
        super().__init__()
        activation = activation or config.activation

        # First calculate the target size after chunking (down projection input size)
        down_size = int((4 / 3) * config.hidden_size)
        # Double it for up projection to ensure chunks match
        up_size = 2 * down_size

        self.up = nn.Linear(config.hidden_size, up_size)
        self.act = ACT2CLS[activation](*args, **kwargs)
        self.dropout = nn.Dropout(config.dropout)
        self.down = nn.Linear(down_size, config.hidden_size)

    def forward(self, inputs, *args, **kwargs):
        a, b = self.up(inputs).chunk(2, dim=-1)
        return self.down(self.dropout(a * self.act(b)))


class PraxisPoly(nn.Module):
    """
    A novel dense layer based on explicit polynomial feature expansions.
    Learns combinations of features up to a specified degree while maintaining
    computational efficiency through careful parameter sharing. Polynomial
    functions are universal approximators (Stone-Weierstrass theorem).
    """

    __version__ = "0.1.0"

    def __init__(
        self,
        config: "AutoConfig",
        degree: int = 6,
        bottleneck: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__()
        dim = config.hidden_size
        self.dim = dim
        self.degree = degree
        self.reduced_dim = int(dim * bottleneck)

        # Reduce dimension for efficiency
        self.down = nn.Linear(dim, self.reduced_dim)

        # Learnable coefficients for each degree
        self.degree_coeffs = nn.ParameterList(
            [nn.Parameter(torch.randn(self.reduced_dim) * 0.02) for _ in range(degree)]
        )

        # Learnable mixing matrices for cross-terms
        self.cross_terms = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.reduced_dim, self.reduced_dim) * 0.02)
                for _ in range(degree - 1)
            ]
        )

        # Project back to original dimension
        self.up = nn.Linear(self.reduced_dim, dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs, *args, **kwargs):
        residual = inputs

        # Project to lower dimension
        x = self.down(inputs)

        # Compute powers and cross-terms
        terms = []
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


class PraxisScatter(nn.Module):
    def __init__(
        self,
        config: "AutoConfig",
        activation=None,
        input_dim=None,
        hidden_dim=None,
        top_k: int = 256,
        depth: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.depth = depth

        # Initialize dimensions
        self.activation = activation or config.activation
        self.input_dim = input_dim or config.hidden_size
        self.hidden_dim = hidden_dim or self.input_dim * 4
        self.top_k = min(top_k, self.hidden_dim)

        # Create multiple input and output projections
        self.input_projections = nn.ModuleList(
            [nn.Linear(self.input_dim, self.hidden_dim) for _ in range(depth)]
        )

        self.output_projections = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.input_dim) for _ in range(depth)]
        )

        # Activation and dropout
        self.act = ACT2FN[self.activation]
        self.dropout = nn.Dropout(config.dropout)

        # Create gate networks - one for each layer except the first
        self.gate_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in range(depth - 1)  # One less than depth
            ]
        )

    def forward(
        self, inputs: torch.Tensor, current_depth: int, *args, **kwargs
    ) -> torch.Tensor:
        """Forward pass with weight and bias sharing through hidden dimension selection."""
        if not 0 <= current_depth < self.depth:
            raise ValueError(f"current_depth must be between 0 and {self.depth-1}")

        # First layer uses original weights and biases
        if current_depth == 0:
            h = self.input_projections[0](inputs)
        else:
            # Get modified weights and biases for current layer
            mod_weights, mod_bias = self.get_modified_weights(
                inputs, current_depth - 1, current_depth
            )

            # Manual linear transformation with modified weights
            h = torch.matmul(inputs, mod_weights.t())

            # Add bias if it exists
            if mod_bias is not None:
                h = h + mod_bias

        h = self.act(h)
        h = self.dropout(h)
        return self.output_projections[current_depth](h)

    def get_modified_weights(
        self, x: torch.Tensor, prev_depth: int, curr_depth: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Helper function to create modified weights and biases for current layer."""
        # Use the appropriate gate network for this depth
        gate_idx = curr_depth - 1  # Adjust index since we have one less gate
        scores = self.gate_networks[gate_idx](x)  # [batch_size, hidden_dim]

        # Get top-k indices from the hidden dimension
        _, top_indices = torch.topk(scores, k=self.top_k, dim=1)  # [batch_size, top_k]
        top_indices = top_indices[0]  # Use first batch's indices [top_k]

        # Get weights
        prev_layer = self.input_projections[prev_depth]
        curr_layer = self.input_projections[curr_depth]

        # Modify weights
        mod_weights = curr_layer.weight.clone()
        mod_weights[top_indices] = prev_layer.weight[top_indices]

        # Handle biases if they exist
        mod_bias = None
        if hasattr(curr_layer, "bias") and curr_layer.bias is not None:
            if hasattr(prev_layer, "bias") and prev_layer.bias is not None:
                mod_bias = curr_layer.bias.clone()
                mod_bias[top_indices] = prev_layer.bias[top_indices]

        return mod_weights, mod_bias
