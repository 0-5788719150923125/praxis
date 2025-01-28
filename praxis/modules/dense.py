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
        top_k: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Initialize dimensions
        self.input_dim = input_dim or config.hidden_size
        self.hidden_dim = hidden_dim or self.input_dim * 4
        self.top_k = top_k or self.hidden_dim // 4

        # Main projection layers
        self.up = nn.Linear(self.input_dim, self.hidden_dim)
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        # Additional weights to pull from
        self.mod = nn.Linear(self.input_dim, self.hidden_dim)
        # Activation and dropout
        self.activation = activation or config.activation
        self.act = ACT2FN[self.activation]
        self.dropout = nn.Dropout(config.dropout)
        self.down = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Ensure input is 3D [batch, seq, features]
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(1)

        # Get modified weights
        mod_weights, mod_bias = self.get_modified_weights(inputs)

        # Apply modified weights
        h = torch.matmul(inputs, mod_weights.transpose(1, 2))
        if mod_bias is not None:
            h = h + mod_bias.unsqueeze(1)

        # Apply activation and dropout
        h = self.act(h)
        h = self.dropout(h)

        # Final projection
        return self.down(h)

    def get_modified_weights(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Handle input dimensions
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(1)  # [batch, 1, input_dim]
        batch_size, seq_len, _ = inputs.shape

        # Process through gate network
        scores = self.gate(inputs)  # [batch, seq, hidden_dim]

        # Flatten and get top-k * seq_len indices
        flat_scores = scores.view(batch_size, -1)  # Using view instead of reshape
        k = min(self.top_k * seq_len, seq_len * self.hidden_dim)
        _, flat_indices = torch.topk(flat_scores, k=k, dim=-1)

        # Convert to hidden dim indices
        hidden_indices = flat_indices % self.hidden_dim

        # Create per-batch weights
        mod_weights = self.up.weight.repeat(batch_size, 1, 1)

        # Create batch indices for scattering
        batch_indices = (
            torch.arange(batch_size, device=inputs.device).unsqueeze(1).expand(-1, k)
        )

        # Update weights using selected indices
        mod_weights[batch_indices, hidden_indices] = self.mod.weight[hidden_indices]

        # Handle biases
        mod_bias = None
        if self.up.bias is not None and self.mod.bias is not None:
            mod_bias = self.up.bias.repeat(batch_size, 1)
            mod_bias[batch_indices, hidden_indices] = self.mod.bias[hidden_indices]

        return mod_weights, mod_bias


# class PraxisScatter(nn.Module):
#     def __init__(
#         self,
#         config: "AutoConfig",
#         activation=None,
#         input_dim=None,
#         hidden_dim=None,
#         top_k: int = None,
#         depth: int = None,
#         *args,
#         **kwargs,
#     ):
#         super().__init__()
#         self.depth = depth or config.depth

#         # Initialize dimensions
#         self.activation = activation or config.activation
#         self.input_dim = input_dim or config.hidden_size
#         self.hidden_dim = hidden_dim or self.input_dim * 4
#         self.top_k = top_k or self.input_dim // 4

#         # Create multiple input and output projections
#         self.up = nn.ModuleList(
#             [nn.Linear(self.input_dim, self.hidden_dim) for _ in range(self.depth)]
#         )

#         self.down = nn.ModuleList(
#             [nn.Linear(self.hidden_dim, self.input_dim) for _ in range(self.depth)]
#         )

#         # Activation and dropout
#         self.act = ACT2FN[self.activation]
#         self.dropout = nn.Dropout(config.dropout)

#         # Create gate networks - one for each layer except the first
#         self.gates = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Linear(self.input_dim, self.hidden_dim),
#                     nn.ReLU(),
#                     nn.Linear(self.hidden_dim, self.hidden_dim),
#                 )
#                 for _ in range(self.depth - 1)  # One less than depth
#             ]
#         )

#     def forward(
#         self, inputs: torch.Tensor, current_depth: int, *args, **kwargs
#     ) -> torch.Tensor:
#         """Forward pass with per-batch weight modification."""
#         # Ensure input is 3D [batch, seq, features]
#         if len(inputs.shape) == 2:
#             inputs = inputs.unsqueeze(1)

#         if not 0 <= current_depth < self.depth:
#             raise ValueError(f"current_depth must be between 0 and {self.depth-1}")

#         if current_depth == 0:
#             h = self.up[0](inputs)
#         else:
#             mod_weights, mod_bias = self.get_modified_weights(
#                 inputs, current_depth - 1, current_depth
#             )

#             # Maintain 3D structure throughout
#             # [batch, seq, in_dim] @ [batch, in_dim, hidden_dim] -> [batch, seq, hidden_dim]
#             h = torch.matmul(inputs, mod_weights.transpose(1, 2))

#             if mod_bias is not None:
#                 h = h + mod_bias.unsqueeze(1)  # Add bias to each sequence position

#         h = self.act(h)
#         h = self.dropout(h)
#         return self.down[current_depth](h)

#     def get_modified_weights(
#         self, x: torch.Tensor, prev_depth: int, curr_depth: int
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         # Handle input dimensions
#         if len(x.shape) == 2:
#             x = x.unsqueeze(1)  # [batch, 1, input_dim]
#         batch_size, seq_len, _ = x.shape

#         # Process through gate network
#         gate_idx = curr_depth - 1
#         scores = self.gates[gate_idx](x)  # [batch, seq, hidden_dim]

#         # Flatten and get top-k * seq_len indices
#         flat_scores = scores.reshape(batch_size, -1)
#         k = min(self.top_k * seq_len, seq_len * self.hidden_dim)
#         _, flat_indices = torch.topk(flat_scores, k=k, dim=-1)

#         # Convert to 2D indices
#         hidden_indices = flat_indices % self.hidden_dim

#         # Get layers
#         prev_layer = self.up[prev_depth]
#         curr_layer = self.up[curr_depth]

#         # Create per-batch weights
#         mod_weights = curr_layer.weight.repeat(batch_size, 1, 1)

#         # Create batch indices for scattering
#         batch_indices = (
#             torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, k)
#         )

#         # Update weights using selected indices
#         mod_weights[batch_indices, hidden_indices] = prev_layer.weight[hidden_indices]

#         # Handle biases
#         mod_bias = None
#         if hasattr(curr_layer, "bias") and curr_layer.bias is not None:
#             if hasattr(prev_layer, "bias") and prev_layer.bias is not None:
#                 mod_bias = curr_layer.bias.repeat(batch_size, 1)
#                 mod_bias[batch_indices, hidden_indices] = prev_layer.bias[
#                     hidden_indices
#                 ]

#         return mod_weights, mod_bias
