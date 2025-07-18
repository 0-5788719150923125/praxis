import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def layernorm_scaling(x, depth, eps=1e-5):
    """
    Apply LayerNorm with depth-based scaling to address the Curse of Depth.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim]
        depth (int): The depth/position of the current layer (starting from 1)
        eps (float): Small constant for numerical stability

    Returns:
        torch.Tensor: Normalized and scaled tensor
    """
    # Ensure depth is at least 1
    depth = max(1, depth)

    # Standard LayerNorm: normalize along the last dimension
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    normalized_x = (x - mean) / torch.sqrt(var + eps)

    # Apply depth-based scaling: divide by sqrt(depth)
    depth_scaling_factor = 1.0 / math.sqrt(depth)
    scaled_x = normalized_x * depth_scaling_factor

    return scaled_x


class LayerNormScaling(nn.Module):
    """
    LayerNorm with depth-based scaling to address the Curse of Depth.
    https://arxiv.org/abs/2502.05795
    This normalizes the input and scales the output by 1/sqrt(depth).
    """

    def __init__(self, normalized_shape, depth, eps=1e-5):
        """
        Args:
            normalized_shape (int or list): Input shape from an expected input
            depth (int): The depth/position of the current layer (starting from 1)
            eps (float): Small constant for numerical stability
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.depth = max(1, depth)  # Ensure depth is at least 1
        self.eps = eps

        # Scaling factor based on depth
        self.register_buffer(
            "depth_scaling_factor",
            torch.tensor(1.0 / math.sqrt(self.depth), dtype=torch.float32),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim]

        Returns:
            torch.Tensor: Normalized and scaled tensor
        """
        # Standard LayerNorm calculation
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)

        # Apply depth-based scaling
        scaled_x = normalized_x * self.depth_scaling_factor

        return scaled_x


def test_layernorm_scaling():
    # Test parameters
    batch_size = 2
    seq_len = 10
    hidden_dim = 64

    # Create random input tensor
    torch.manual_seed(42)  # For reproducibility
    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Test with different depths
    depths = [1, 2, 4, 16]

    for depth in depths:
        # Apply layernorm_scaling
        scaled_output = layernorm_scaling(x, depth)

        # Calculate expected scaling factor
        expected_scale_factor = 1.0 / math.sqrt(depth)

        # Verify shape
        assert (
            scaled_output.shape == x.shape
        ), f"Output shape mismatch for depth {depth}"

        # Verify normalized values (should have mean≈0, var≈expected_scale_factor²)
        mean = scaled_output.mean(dim=-1)
        var = scaled_output.var(dim=-1, unbiased=False)

        # Mean should be close to 0
        assert torch.allclose(
            mean, torch.zeros_like(mean), atol=1e-6
        ), f"Mean not close to 0 for depth {depth}"

        # Variance should be close to expected_scale_factor²
        expected_var = expected_scale_factor * expected_scale_factor
        assert torch.allclose(
            var, torch.ones_like(var) * expected_var, atol=1e-5
        ), f"Variance not scaled correctly for depth {depth}"

        print(
            f"Depth {depth}: scaling factor = {expected_scale_factor:.4f}, "
            f"mean = {mean.mean().item():.6f}, "
            f"variance = {var.mean().item():.6f} (expected ≈ {expected_var:.6f})"
        )

    print("All tests passed!")

    # Also test the nn.Module implementation
    print("\nTesting LayerNormScaling class:")
    for depth in depths:
        ln_scaling = LayerNormScaling(hidden_dim, depth)
        scaled_output = ln_scaling(x)

        # Calculate expected scaling factor
        expected_scale_factor = 1.0 / math.sqrt(depth)
        expected_var = expected_scale_factor * expected_scale_factor

        # Verify stats
        mean = scaled_output.mean(dim=-1)
        var = scaled_output.var(dim=-1, unbiased=False)

        print(
            f"Depth {depth}: scaling factor = {expected_scale_factor:.4f}, "
            f"mean = {mean.mean().item():.6f}, "
            f"variance = {var.mean().item():.6f} (expected ≈ {expected_var:.6f})"
        )


if __name__ == "__main__":
    test_layernorm_scaling()
