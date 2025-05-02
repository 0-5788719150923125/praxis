import math
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor

from praxis.activations import ACT2CLS, ACT2FN

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class ScatterMLP(nn.Module):
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
