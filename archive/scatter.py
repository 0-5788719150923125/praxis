import math
from collections import OrderedDict
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
        depth: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.depth = depth or config.depth

        # Initialize dimensions
        self.activation = activation or config.activation
        self.input_dim = input_dim or config.hidden_size
        self.hidden_dim = hidden_dim or self.input_dim * 4
        # Change top_k calculation to use hidden_dim
        self.top_k = top_k or self.hidden_dim // 4  # Now relative to hidden_dim

        # Create multiple input and output projections
        self.up = nn.ModuleList(
            [nn.Linear(self.input_dim, self.hidden_dim) for _ in range(self.depth)]
        )

        self.down = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.input_dim) for _ in range(self.depth)]
        )

        # Activation and dropout
        self.act = ACT2FN[self.activation]
        self.dropout = nn.Dropout(config.dropout)

        # Create gate networks - one for each layer except the first
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in range(self.depth - 1)  # One less than depth
            ]
        )

    def forward(
        self, inputs: torch.Tensor, current_depth: int, *args, **kwargs
    ) -> torch.Tensor:
        """Forward pass with per-batch weight modification."""
        # Ensure input is 3D [batch, seq, features]
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(1)

        if not 0 <= current_depth < self.depth:
            raise ValueError(f"current_depth must be between 0 and {self.depth-1}")

        if current_depth == 0:
            h = self.up[0](inputs)
        else:
            mod_weights, mod_bias = self.get_modified_weights(
                inputs, current_depth - 1, current_depth
            )

            # Maintain 3D structure throughout
            # [batch, seq, in_dim] @ [batch, in_dim, hidden_dim] -> [batch, seq, hidden_dim]
            h = torch.matmul(inputs, mod_weights.transpose(1, 2))

            if mod_bias is not None:
                h = h + mod_bias.unsqueeze(1)  # Add bias to each sequence position

        h = self.act(h)
        h = self.dropout(h)
        return self.down[current_depth](h)

    def get_modified_weights(
        self, x: torch.Tensor, prev_depth: int, curr_depth: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Handle input dimensions
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
        batch_size, seq_len, _ = x.shape

        # Process through gate network
        gate_idx = curr_depth - 1
        scores = self.gates[gate_idx](x)  # [batch, seq, hidden_dim]

        # Flatten and get top-k * seq_len indices
        flat_scores = scores.reshape(batch_size, -1)
        k = min(self.top_k * seq_len, seq_len * self.hidden_dim)
        _, flat_indices = torch.topk(flat_scores, k=k, dim=-1)

        # Convert to 2D indices
        hidden_indices = flat_indices % self.hidden_dim

        # Get layers
        prev_layer = self.up[prev_depth]
        curr_layer = self.up[curr_depth]

        # Create per-batch weights
        mod_weights = curr_layer.weight.repeat(batch_size, 1, 1)

        # Create batch indices for scattering
        batch_indices = (
            torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, k)
        )

        # Update weights using selected indices
        mod_weights[batch_indices, hidden_indices] = prev_layer.weight[hidden_indices]

        # Handle biases
        mod_bias = None
        if hasattr(curr_layer, "bias") and curr_layer.bias is not None:
            if hasattr(prev_layer, "bias") and prev_layer.bias is not None:
                mod_bias = curr_layer.bias.repeat(batch_size, 1)
                mod_bias[batch_indices, hidden_indices] = prev_layer.bias[
                    hidden_indices
                ]

        return mod_weights, mod_bias


if __name__ == "__main__":
    # Test the implementation
    class DummyConfig:
        activation = "gelu"
        hidden_size = 64
        dropout = 0.1

    config = DummyConfig()

    # Create model
    model = PraxisScatter(config=config, depth=3)

    # Test forward pass at different depths
    batch_size = 4
    x = torch.randn(batch_size, config.hidden_size)

    print("\nTesting forward pass at different depths:")
    for depth in range(3):
        print(f"\nDepth {depth}:")

        if depth > 0:
            prev_layer = model.up[depth - 1]
            curr_layer = model.up[depth]

            # Store original weights and compute initial statistics
            orig_weights = curr_layer.weight.clone()
            orig_bias = curr_layer.bias.clone() if curr_layer.bias is not None else None

            # Do forward pass (which modifies weights)
            out = model(x, current_depth=depth)
            print(f"Output shape: {out.shape}")

            # Get modified weights
            mod_weights, mod_bias = model.get_modified_weights(x, depth - 1, depth)

            # Compute weight statistics
            weight_diff = mod_weights - orig_weights
            print(f"Weight modification statistics:")
            print(f"  Mean diff: {weight_diff.mean().item():.6f}")
            print(f"  Std diff: {weight_diff.std().item():.6f}")
            print(f"  L2 norm of diff: {torch.norm(weight_diff).item():.6f}")

            # Compute proportion of weights modified
            total_weights = weight_diff.numel()
            modified_weights = torch.count_nonzero(weight_diff)
            print(
                f"  Modified {modified_weights.item()/total_weights*100:.2f}% of weights"
            )

            if orig_bias is not None and mod_bias is not None:
                bias_diff = mod_bias - orig_bias
                print(f"\nBias modification statistics:")
                print(f"  Mean diff: {bias_diff.mean().item():.6f}")
                print(f"  Std diff: {bias_diff.std().item():.6f}")
                print(f"  L2 norm of diff: {torch.norm(bias_diff).item():.6f}")

                total_biases = bias_diff.numel()
                modified_biases = torch.count_nonzero(bias_diff)
                print(
                    f"  Modified {modified_biases.item()/total_biases*100:.2f}% of biases"
                )
        else:
            out = model(x, current_depth=depth)
            print(f"Output shape: {out.shape}")
