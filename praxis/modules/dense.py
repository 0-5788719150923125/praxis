import math
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor

from praxis.activations import ACT2CLS, ACT2FN

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class PraxisMLP(nn.Sequential):
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


class PraxisGLU(nn.Module):
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


class PraxisScatter(nn.Module):
    """
    A scatter-based neural network layer that dynamically modifies weights
    during the forward pass based on input features.
    """

    def __init__(
        self,
        config: ConfigType,
        activation: Optional[str] = None,
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        top_k: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a scatter-based layer.

        Args:
            config: Configuration object with model parameters
            activation: Activation function name (from ACT2FN registry)
            input_dim: Input dimension size
            hidden_dim: Hidden dimension size
            top_k: Number of top weights to modify
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__()

        # Initialize dimensions
        self.input_dim: int = input_dim or config.hidden_size
        self.hidden_dim: int = hidden_dim or self.input_dim * 4
        self.top_k: int = top_k or self.hidden_dim // 4

        # Main projection layers
        self.up: nn.Linear = nn.Linear(self.input_dim, self.hidden_dim)
        # Gate network
        self.gate: nn.Sequential = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        # Additional weights to pull from
        self.mod: nn.Linear = nn.Linear(self.input_dim, self.hidden_dim)
        # Activation and dropout
        self.activation: str = activation or config.activation
        self.act: Callable[[Tensor], Tensor] = ACT2FN[self.activation]
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)
        self.down: nn.Linear = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        Forward pass through the scatter layer.

        Args:
            inputs: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after scatter processing
        """
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

    def get_modified_weights(self, inputs: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Compute dynamically modified weights based on input features.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            Tuple containing:
                - Modified weight tensor of shape [batch_size, hidden_dim, input_dim]
                - Optional modified bias tensor of shape [batch_size, hidden_dim]
        """
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
        mod_bias: Optional[Tensor] = None
        if self.up.bias is not None and self.mod.bias is not None:
            mod_bias = self.up.bias.repeat(batch_size, 1)
            mod_bias[batch_indices, hidden_indices] = self.mod.bias[hidden_indices]

        return mod_weights, mod_bias
