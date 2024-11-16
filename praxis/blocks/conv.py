import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.activations import ACT2FN
from praxis.modules.dense import PraxisGLU


class PraxisConv(nn.Module):
    """
    A special kind of block that omits the self-attention mechanism, in favor
    of causal convolutional layers and periodic activations. While this was originally
    inspired by NanoFFT, the module looks almost nothing like that now.
    https://github.com/timurgepard/nanoFFT
    Periodic activation functions can train a model to "know what they do not know."
    https://arxiv.org/abs/2110.13572
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig", *args, **kwargs):
        super().__init__()
        hidden_dim = config.num_dims

        # Local processing
        self.conv_norm = nn.LayerNorm(hidden_dim)
        self.conv = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3)

        # Global context processing
        self.gc = CausalGlobalContext(hidden_dim, capacity=0.75)

        config.activation = "sin_cos"
        self.ffw_norm = nn.LayerNorm(hidden_dim)
        self.ffw = PraxisGLU(config)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        router_weights: Optional[Tensor] = None,
        token_indices: Optional[Tensor] = None,
    ) -> Tensor:
        # Local processing
        residual = x
        x_norm = self.conv_norm(x)
        x_transposed = x_norm.transpose(1, 2)  # (B, E, T)
        x_conv = self.conv(x_transposed)

        # Global context
        x_gc = self.gc(x_conv)

        # Back to sequence format
        x_out = x_gc.transpose(1, 2)  # (B, T, E)

        # Residual
        residual = x_out + residual

        # FFN
        x_norm = self.ffw_norm(residual)
        x_ffw = self.ffw(x_norm)
        return x_ffw + residual


class CausalConv1d(nn.Conv1d):
    """1D Causal Convolution Layer."""

    def __init__(
        self, in_channels, out_channels, kernel_size, dilation=1, bias=False, **kwargs
    ):
        padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            bias=bias,
            **kwargs,
        )
        self.left_padding = padding

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return super().forward(x)


class CausalGlobalContext(nn.Module):
    """
    Implements a kind of squeeze-and-excitation mechanism, which allows
    us to bridge convolutional operations' local contexts, into a global one.
    https://arxiv.org/abs/1904.11492v1
    """

    def __init__(self, in_channels, capacity=0.125):
        super().__init__()
        bottleneck = int(in_channels * capacity)

        # Context modeling - single 1x1 conv to generate global attention weights
        self.context = nn.Conv1d(in_channels, 1, kernel_size=1)

        # Bottleneck transform with Conv1d layers
        self.transform = nn.Sequential(
            # First conv reduces channels
            nn.Conv1d(in_channels, bottleneck, kernel_size=1),
            # LayerNorm needs to be applied to channel dim for conv
            nn.GroupNorm(1, bottleneck),  # equivalent to LayerNorm for conv
            ACT2FN["periodic_relu"],
            # Second conv restores channels
            nn.Conv1d(bottleneck, in_channels, kernel_size=1),
        )

        # Two learnable parameters for position bias
        self.pos_bias_start = nn.Parameter(torch.tensor([0.1]))
        self.pos_bias_end = nn.Parameter(torch.tensor([-0.1]))

    def forward(self, x):
        B, C, T = x.shape

        # Generate attention weights
        weights = self.context(x)  # B, 1, T

        # Apply causal masking
        mask = torch.triu(torch.ones_like(weights), diagonal=1)
        weights = weights.masked_fill(mask.bool(), float("-inf"))

        # Create position-aware bias with learned start and end values
        positions = torch.linspace(0, 1, T, device=x.device)
        position_bias = (
            self.pos_bias_start + (self.pos_bias_end - self.pos_bias_start) * positions
        )
        position_bias = position_bias.view(1, 1, -1)  # B, 1, T

        # Add position bias to masked scores before softmax
        weights = weights + position_bias

        # Apply softmax
        scores = F.softmax(weights, dim=-1)  # B, 1, T

        # Calculate global context
        context = torch.matmul(x, scores.transpose(-2, -1))  # B, C, 1

        # Transform through bottleneck (no need to squeeze/unsqueeze)
        context = self.transform(context)  # B, C, 1

        # Broadcast and add to input
        return x + context.expand(-1, -1, x.size(2))


if __name__ == "__main__":
    from dataclasses import dataclass

    # Mock AutoConfig class to simulate the configuration
    @dataclass
    class AutoConfig:
        num_dims: int = 768
        num_embeds: int = 768
        context_length: int = 2048
        vocab_size: int = 50257
        causal: bool = True
        dropout: float = 0.0

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig()
    chunk_size = 256  # Explicitly define for tests
    stride = 128  # Example stride with overlap

    def run_memory_test(model, x):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Forward pass
        start_time = time.time()
        with torch.cuda.amp.autocast():
            output = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        max_memory = (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
        )  # Convert to MB
        return output, end_time - start_time, max_memory

    print("Running tests for PraxisConv...")

    # Create model once for all tests
    model = PraxisConv(config, chunk_size=chunk_size, stride=stride).to(device)

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
        model_test = PraxisConv(config, chunk_size=cs, stride=current_stride).to(device)
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

    print("\nAll tests completed!")
