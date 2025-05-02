import math
from typing import Optional, Tuple, TypeVar

import torch
import torch.nn as nn

from praxis.activations import ACT2CLS, ACT2FN

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class PraxisScatter(nn.Module):
    def __init__(
        self,
        config: ConfigType,
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
        self.down = nn.Linear(self.hidden_dim, self.input_dim)

        # Additional weights to pull from
        self.mod = nn.Linear(self.input_dim, self.hidden_dim)

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Activation and dropout
        self.activation = activation or config.activation
        self.act = ACT2FN[self.activation]
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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


if __name__ == "__main__":
    # Test configuration
    class DummyConfig:
        activation = "gelu"
        hidden_size = 64
        dropout = 0.1

    config = DummyConfig()

    # Create model
    model = PraxisScatter(config=config)

    # Test inputs
    batch_size = 4
    seq_len = 3
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    # Store original weights
    orig_weights = model.up.weight.clone()
    orig_bias = model.up.bias.clone() if model.up.bias is not None else None

    # Forward pass
    out = model(x)
    print(f"\nOutput shape: {out.shape}")

    # Get modified weights for analysis
    mod_weights, mod_bias = model.get_modified_weights(x)

    # Compute statistics
    weight_diff = mod_weights - orig_weights
    print(f"\nWeight modification statistics:")
    print(f"  Mean diff: {weight_diff.mean().item():.6f}")
    print(f"  Std diff: {weight_diff.std().item():.6f}")
    print(f"  L2 norm of diff: {torch.norm(weight_diff).item():.6f}")

    # Compute proportion of weights modified
    total_weights = weight_diff.numel() // batch_size  # Per batch
    modified_weights = torch.count_nonzero(weight_diff[0])  # First batch
    print(f"  Modified {modified_weights.item()/total_weights*100:.2f}% of weights")

    if orig_bias is not None and mod_bias is not None:
        bias_diff = mod_bias - orig_bias
        print(f"\nBias modification statistics:")
        print(f"  Mean diff: {bias_diff.mean().item():.6f}")
        print(f"  Std diff: {bias_diff.std().item():.6f}")
        print(f"  L2 norm of diff: {torch.norm(bias_diff).item():.6f}")

        total_biases = bias_diff.numel() // batch_size  # Per batch
        modified_biases = torch.count_nonzero(bias_diff[0])  # First batch
        print(f"  Modified {modified_biases.item()/total_biases*100:.2f}% of biases")
