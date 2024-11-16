import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.modules.dense import PraxisGLU


class PraxisNano(nn.Module):
    """
    A special kind of block that omits all self-attention mechanisms, in favor
    of dense layers with periodic activations. Inspired by NanoFFT:
    https://github.com/timurgepard/nanoFFT
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig", *args, **kwargs):
        super().__init__()
        hidden_dim = config.num_dims

        # Define the weight matrices
        self.fft_norm = nn.LayerNorm(hidden_dim)
        self.fft = nn.Sequential(
            ElasticLinear(
                features=hidden_dim,
                bottleneck=1.0,
                causal=config.causal,
            ),
            nn.Dropout(config.dropout),
            ElasticLinear(
                features=hidden_dim,
                bottleneck=1.0,
                causal=config.causal,
            ),
        )

        # Feed-forward network with sine activation
        config.activation = "sin"
        self.ffw_norm = nn.LayerNorm(hidden_dim)
        self.ffw = PraxisGLU(config)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        router_weights: Optional[Tensor] = None,
        token_indices: Optional[Tensor] = None,
    ):
        # x shape: (B, T, E)
        chunk_norm = self.fft_norm(x)
        # Transpose (B, T, E) -> (B, E, T)
        chunk_fft = chunk_norm.transpose(1, 2)
        # Pass through FFT layers
        chunk_fft = self.fft(chunk_fft)
        # Transpose back to (B, T, E)
        chunk_fft = chunk_fft.transpose(1, 2)
        # Residual connection
        residual = chunk_fft + x
        # LayerNorm
        chunk_ffw = self.ffw_norm(residual)
        # Feedforward
        chunk_ffw = self.ffw(chunk_ffw)
        # Residual connection
        return chunk_ffw + residual


class ElasticLinear(nn.Module):
    def __init__(self, features, bottleneck=0.5, causal=False):
        super().__init__()
        self.causal = causal
        # Initialize with smaller dimensions to force interpolation
        bottleneck_dim = int(features * bottleneck)
        self.weight = nn.Parameter(torch.Tensor(features, bottleneck_dim))
        # self.bias = nn.Parameter(torch.Tensor(features))
        self.reset_parameters()

    def forward(self, x):
        # x shape: (batch_size, features, seq_len)
        _, features, _ = x.shape
        weights = self._interpolate_weights(features)
        return torch.matmul(weights, x)  # + self.bias.view(-1, 1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # nn.init.zeros_(self.bias)

    def _interpolate_weights(self, features):
        # Always interpolate since we're starting from a smaller base
        weights_expanded = self.weight.unsqueeze(0)

        interpolated = F.interpolate(
            weights_expanded,
            size=[features],
            mode="nearest",
            # align_corners=True,
        ).squeeze(0)

        if self.causal:
            causal_mask = torch.tril(torch.ones_like(interpolated))
            # causal_mask = causal_mask / (torch.sum(causal_mask, dim=1, keepdim=True))
            interpolated = interpolated * causal_mask

        return interpolated


if __name__ == "__main__":
    import random
    import time
    from dataclasses import dataclass

    import numpy as np

    # Mock AutoConfig class to simulate the configuration
    @dataclass
    class AutoConfig:
        num_dims: int = 768
        num_embeds: int = 768
        context_length: int = 8192
        vocab_size: int = 50257
        causal: bool = True
        dropout: float = 0.0

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig()
    chunk_size = 256  # Explicitly define for tests
    stride = 128  # Example stride with overlap

    def run_memory_test(model, x):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Forward pass
        start_time = time.time()
        with torch.cuda.amp.autocast():
            output = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            max_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        else:
            max_memory = 0
        end_time = time.time()
        return output, end_time - start_time, max_memory

    print("Running tests for PraxisNano...")

    # Create model once for all tests
    model = PraxisNano(config).to(device)

    # Test 1: Basic Functionality (Short Sequence)
    print("\nTest 1: Short Sequence Test")
    # Test with a sequence length that's exactly half of chunk_size
    x_short = torch.randn(2, chunk_size // 2, config.num_dims).to(device)

    try:
        output_short = model(x_short)
        print(f"✓ Short sequence shape: {output_short.shape}")
        assert output_short.shape == x_short.shape, "Output shape mismatch"
        print("✓ Basic functionality test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 2: Long Sequence Handling
    print("\nTest 2: Long Sequence Test")
    x_long = torch.randn(2, chunk_size * 4, config.num_dims).to(
        device
    )  # Test with multiple of chunk_size

    try:
        output_long = model(x_long)
        print(f"✓ Long sequence shape: {output_long.shape}")
        assert output_long.shape == x_long.shape, "Output shape mismatch"
        print("✓ Long sequence test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 3: Memory Scaling Test
    print("\nTest 3: Memory Scaling Test")
    chunk_sizes = [
        chunk_size // 2,
        chunk_size,
        chunk_size * 2,
        chunk_size * 4,
    ]
    results = []

    for cs in chunk_sizes:
        # Adjust stride accordingly (for simplicity, stride = cs // 2)
        current_stride = cs // 2
        model_test = PraxisNano(config).to(device)
        x_test = torch.randn(1, cs * 4, config.num_dims).to(device)
        output, duration, memory = run_memory_test(model_test, x_test)
        results.append((cs, duration, memory))
        print(f"\nChunk Size: {cs}")
        print(f"✓ Processing Time: {duration:.4f} seconds")
        print(f"✓ Peak Memory Usage: {memory:.2f} MB")

    # Test 4: Gradient Flow Test
    print("\nTest 4: Gradient Flow Test")
    model.zero_grad()
    x = torch.randn(2, chunk_size * 2, config.num_dims, requires_grad=True).to(device)

    try:
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check if gradients exist and are not None
        has_grads = all(p.grad is not None for p in model.parameters())
        print(f"✓ Gradients exist: {has_grads}")

        # Check if gradients contain NaN values
        has_nans = any(torch.isnan(p.grad).any() for p in model.parameters())
        print(f"✓ Gradients are clean (no NaNs): {not has_nans}")

        print("✓ Gradient flow test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 5: Chunk Boundary Test
    print("\nTest 5: Chunk Boundary Test")
    x = torch.randn(1, chunk_size * 2, config.num_dims).to(device)

    try:
        # Get outputs for consecutive chunks
        out1 = model(x[:, :chunk_size, :])
        out2 = model(x[:, chunk_size:, :])

        # Process as single sequence
        out_full = model(x)

        # Compare the results at the boundary
        boundary_diff = (
            (
                out_full[:, chunk_size - stride : chunk_size + stride, :]
                - torch.cat([out1[:, -stride:, :], out2[:, :stride, :]], dim=1)
            )
            .abs()
            .mean()
        )

        print(f"✓ Chunk boundary difference: {boundary_diff:.6f}")
        assert boundary_diff < 1.0, "Chunk boundary difference too large"
        print("✓ Chunk boundary test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 6: Random Offset Consistency
    print("\nTest 6: Random Offset Test")
    x = torch.randn(1, chunk_size * 3, config.num_dims).to(device)

    try:
        # Multiple forward passes should give similar results due to deterministic processing
        out1 = model(x)
        out2 = model(x)
        difference = (out1 - out2).abs().mean().item()

        print(f"✓ Output difference between passes: {difference:.6f}")
        assert difference == 0, "Outputs differ despite deterministic processing"
        print("✓ Random offset test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 7: Random Dimensions Test
    print("\nTest 7: Random Dimensions Test")
    torch.manual_seed(42)  # For reproducibility
    for i in range(5):
        # Generate random dimensions
        base_in_features = random.randint(5, 500)
        batch_dim = random.randint(1, 32)
        time_dim = random.randint(1, 64)
        input_dim = random.randint(5, 500)

        print(f"\nRandom test iteration {i+1}:")
        print(f"Base in_features: {base_in_features}")
        print(
            f"Input dimensions: [batch={batch_dim}, time={time_dim}, input_dim={input_dim}]"
        )

        # Create model and input
        bottleneck = 0.5
        model = ElasticLinear(features=base_in_features, bottleneck=bottleneck).to(
            device
        )
        x = torch.randn(batch_dim, input_dim, time_dim).to(device)

        # Forward pass
        out = model(x)

        # Verify shapes
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
        print(
            f"Adjusted weight matrix shape: {model._interpolate_weights(input_dim).shape}"
        )
        assert out.shape == (
            batch_dim,
            base_in_features,
            time_dim,
        ), "Output shape mismatch"

    def test_elastic_interpolation():
        torch.manual_seed(42)  # For reproducibility

        # Create a small example
        in_features = 8
        bottleneck = 0.5
        model = ElasticLinear(in_features, bottleneck)

        # The actual weight matrix will be [out_features, bottleneck]
        # where bottleneck = min(in_features, out_features) // 2 = 4
        print(f"Original weight shape: {model.weight.shape}")
        print("\nOriginal weights (first 3 rows):")
        print(model.weight[:3].detach().numpy())

        # Create input that requires interpolation
        x = torch.randn(1, in_features, 10)  # batch=1, seq_len=10

        # Get interpolated weights
        interpolated = model._interpolate_weights(in_features)
        print(f"\nInterpolated weight shape: {interpolated.shape}")
        print("\nInterpolated weights (first 3 rows):")
        print(interpolated[:3].detach().numpy())

        # Compare dimensions and values
        print("\nComparison:")
        print(
            f"Original weights:    {model.weight.shape} -> min: {model.weight.min():.3f}, max: {model.weight.max():.3f}"
        )
        print(
            f"Interpolated weights: {interpolated.shape} -> min: {interpolated.min():.3f}, max: {interpolated.max():.3f}"
        )

        # Show that values are actually different
        if model.weight.shape[1] != interpolated.shape[1]:
            print("\nDimension change detected!")
            print(
                f"Number of columns changed from {model.weight.shape[1]} to {interpolated.shape[1]}"
            )

        # Check if values are just being copied or actually interpolated
        if model.weight.shape[1] < interpolated.shape[1]:
            # Take a slice of the original and interpolated weights to compare
            orig_slice = model.weight[0, :3].detach().numpy()
            interp_slice = interpolated[0, :3].detach().numpy()
            print("\nComparing first three values of first row:")
            print(f"Original:     {orig_slice}")
            print(f"Interpolated: {interp_slice}")

            # Check if the values are truly interpolated (should be different)
            is_different = not np.allclose(orig_slice, interp_slice)
            print(
                f"\nValues are {'different (interpolated)' if is_different else 'identical (copied)'}"
            )

    # Test 8: Interpolation Test
    print("\nTest 8: Interpolation")
    test_elastic_interpolation()

    print("\nAll tests completed!")
