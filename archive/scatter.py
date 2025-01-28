import math
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn

from praxis.activations import ACT2CLS, ACT2FN


class PraxisScatter(nn.Module):
    def __init__(
        self,
        config: "AutoConfig",
        activation=None,
        input_dim=None,
        hidden_dim=None,
        depth: int = 2,
        top_k: int = 256,
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

    def forward(self, x: torch.Tensor, current_depth: int) -> torch.Tensor:
        """Forward pass with weight and bias sharing through hidden dimension selection."""
        if not 0 <= current_depth < self.depth:
            raise ValueError(f"current_depth must be between 0 and {self.depth-1}")

        # First layer uses original weights and biases
        if current_depth == 0:
            h = self.input_projections[0](x)
        else:
            # Get modified weights and biases for current layer
            mod_weights, mod_bias = self.get_modified_weights(
                x, current_depth - 1, current_depth
            )

            # Manual linear transformation with modified weights
            h = torch.matmul(x, mod_weights.t())

            # Add bias if it exists
            if mod_bias is not None:
                h = h + mod_bias

        h = self.act(h)
        h = self.dropout(h)
        return self.output_projections[current_depth](h)


if __name__ == "__main__":
    # Test the implementation
    class DummyConfig:
        activation = "gelu"
        hidden_size = 64
        dropout = 0.1

    config = DummyConfig()

    # Create model
    model = PraxisScatter(config=config, depth=3, top_k=128)

    # Test forward pass at different depths
    batch_size = 4
    x = torch.randn(batch_size, config.hidden_size)

    print("\nTesting forward pass at different depths:")
    for depth in range(3):
        print(f"\nDepth {depth}:")

        if depth > 0:
            prev_layer = model.input_projections[depth - 1]
            curr_layer = model.input_projections[depth]

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
