from typing import Optional, OrderedDict

import torch
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
        **kwargs
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


class PraxisGLU(nn.Module):
    """
    A standard MLP, augmented with a Gated Linear Units.
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

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
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
        **kwargs
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

    def forward(self, x):
        residual = x

        # Project to lower dimension
        x = self.down(x)

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
