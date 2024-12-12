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
    of causal convolutional layers and periodic activations.
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig", *args, **kwargs):
        super().__init__()
        hidden_dim = config.hidden_size
        projection = int(hidden_dim * (1 + config.capacity))

        # Local processing
        self.conv_norm = nn.LayerNorm(hidden_dim)
        # self.conv = CausalConv1d(hidden_dim, projection, kernel_size=7)
        self.conv = MultiHeadCausalConv1d(hidden_dim, projection, num_heads=3)

        # Global context processing
        self.gc = CausalGlobalContext(projection, capacity=config.capacity)
        self.reduce = nn.Linear(projection, hidden_dim)

        config.activation = "sin_cos"
        self.ffw_norm = nn.LayerNorm(hidden_dim)
        self.ffw = PraxisGLU(config)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        router_weights: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        # Local processing
        residual = x
        x_norm = self.conv_norm(x)  # Shape: (B, T, E)
        x_transposed = x_norm.transpose(1, 2)  # Shape: (B, E, T)
        x_conv = self.conv(x_transposed)  # Shape: (B, projection, T)

        # Global context
        x_gc = self.gc(x_conv)  # Shape: (B, projection, T)
        x_gc = x_gc.transpose(1, 2)  # Transpose to (B, T, projection)
        x_out = self.reduce(x_gc)  # Linear layer applied to feature dimension
        # Output shape: (B, T, hidden_dim)

        # Residual connection and FFN
        residual = x_out + residual  # Shape: (B, T, hidden_dim)

        x_norm = self.ffw_norm(residual)
        x_ffw = self.ffw(x_norm)
        return x_ffw + residual


class MultiHeadCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super().__init__()
        assert out_channels % num_heads == 0
        head_dim = out_channels // num_heads
        self.convs = nn.ModuleList(
            [CausalConv1d(in_channels, head_dim, (i + 3)) for i in range(num_heads)]
        )

    def forward(self, x):
        head_outputs = [conv(x) for conv in self.convs]
        return torch.cat(head_outputs, dim=1)


# class MultiHeadCausalConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, num_heads, kernel_size=3):
#         super().__init__()
#         assert out_channels % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = out_channels // num_heads

#         # Single set of convolutions for all heads
#         self.query_conv = nn.Conv1d(
#             in_channels,
#             out_channels,
#             kernel_size=kernel_size,
#             padding=kernel_size - 1,
#             groups=num_heads,
#         )
#         self.key_conv = nn.Conv1d(
#             in_channels,
#             out_channels,
#             kernel_size=kernel_size,
#             padding=kernel_size - 1,
#             groups=num_heads,
#         )
#         self.value_conv = nn.Conv1d(
#             in_channels,
#             out_channels,
#             kernel_size=kernel_size,
#             padding=kernel_size - 1,
#             groups=num_heads,
#         )

#         # Layer normalization for better training stability
#         self.layer_norm = nn.LayerNorm(out_channels)

#         self.scale = (out_channels // num_heads) ** -0.5

#     def forward(self, x):
#         B, C, T = x.size()
#         H = self.num_heads

#         # Apply convolutions and reshape for multi-head attention
#         # Shape: (B, C, T) -> (B, H, D, T) where D = C//H
#         queries = self.query_conv(x)[:, :, -T:]  # Remove padding
#         keys = self.key_conv(x)[:, :, -T:]
#         values = self.value_conv(x)[:, :, -T:]

#         # Reshape to separate heads
#         queries = queries.view(B, H, -1, T)
#         keys = keys.view(B, H, -1, T)
#         values = values.view(B, H, -1, T)

#         # Compute attention scores for all heads simultaneously
#         # (B, H, D, T) @ (B, H, T, D) -> (B, H, T, T)
#         scores = torch.matmul(queries.transpose(2, 3), keys) * self.scale

#         # Causal masking
#         mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
#         scores = scores.masked_fill(mask, float("-inf"))

#         # Apply softmax
#         attn_weights = F.softmax(scores, dim=-1)

#         # Apply attention weights to values
#         # (B, H, T, T) @ (B, H, D, T).transpose(2,3) -> (B, H, T, D)
#         out = torch.matmul(attn_weights, values.transpose(2, 3))

#         # Reshape back
#         out = out.transpose(2, 3).contiguous().view(B, -1, T)

#         # Apply layer normalization
#         out = self.layer_norm(out.transpose(1, 2)).transpose(1, 2)

#         return out


# class MultiHeadCausalConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, num_heads):
#         super().__init__()
#         assert out_channels % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = out_channels // num_heads

#         # Convolutions to compute queries, keys, and values
#         self.query_convs = nn.ModuleList(
#             [
#                 CausalConv1d(in_channels, self.head_dim, kernel_size=(i + 3))
#                 for i in range(num_heads)
#             ]
#         )
#         self.key_convs = nn.ModuleList(
#             [
#                 CausalConv1d(in_channels, self.head_dim, kernel_size=(i + 3))
#                 for i in range(num_heads)
#             ]
#         )
#         self.value_convs = nn.ModuleList(
#             [
#                 CausalConv1d(in_channels, self.head_dim, kernel_size=(i + 3))
#                 for i in range(num_heads)
#             ]
#         )

#         self.scale = self.head_dim**-0.5  # Scaling factor for dot product attention

#     def forward(self, x):
#         # x shape: (B, C, T)
#         B, C, T = x.size()
#         attention_outputs = []

#         for i in range(self.num_heads):
#             # Compute queries, keys, and values
#             Q = self.query_convs[i](x)  # Shape: (B, head_dim, T)
#             K = self.key_convs[i](x)  # Shape: (B, head_dim, T)
#             V = self.value_convs[i](x)  # Shape: (B, head_dim, T)

#             # Compute attention scores
#             # Transpose K to match dimensions for batch matrix multiplication
#             attn_scores = torch.bmm(Q.transpose(1, 2), K)  # Shape: (B, T, T)
#             attn_scores = attn_scores * self.scale

#             # Apply causal masking
#             causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0)
#             attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

#             # Apply softmax to get attention weights
#             attn_weights = F.softmax(attn_scores, dim=-1)  # Shape: (B, T, T)

#             # Compute weighted sum of values
#             attn_output = torch.bmm(
#                 attn_weights, V.transpose(1, 2)
#             )  # Shape: (B, T, head_dim)
#             attn_output = attn_output.transpose(1, 2)  # Shape: (B, head_dim, T)
#             attention_outputs.append(attn_output)

#         # Concatenate the outputs from all heads
#         output = torch.cat(attention_outputs, dim=1)  # Shape: (B, out_channels, T)
#         return output


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
        hidden_size: int = 768
        embed_size: int = 768
        num_heads: int = 4
        context_length: int = 2048
        vocab_size: int = 50257
        causal: bool = True
        dropout: float = 0.0
        capacity: float = 0.125

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
    x_short = torch.randn(2, chunk_size // 2, config.hidden_size).to(device)

    try:
        output_short = model(x_short)
        print(f"✓ Short sequence shape: {output_short.shape}")
        assert output_short.shape == x_short.shape, "Output shape mismatch"
        print("✓ Basic functionality test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 2: Long Sequence Handling
    print("\nTest 2: Long Sequence Test")
    x_long = torch.randn(2, chunk_size * 4, config.hidden_size).to(
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
        x_test = torch.randn(1, cs * 4, config.hidden_size).to(device)
        output, duration, memory = run_memory_test(model_test, x_test)
        results.append((cs, duration, memory))
        print(f"\nChunk Size: {cs}")
        print(f"✓ Processing Time: {duration:.4f} seconds")
        print(f"✓ Peak Memory Usage: {memory:.2f} MB")

    # Test 4: Gradient Flow Test
    print("\nTest 4: Gradient Flow Test")
    model.zero_grad()
    x = torch.randn(2, chunk_size * 2, config.hidden_size, requires_grad=True).to(
        device
    )

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
    x = torch.randn(1, chunk_size * 2, config.hidden_size).to(device)

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
    x = torch.randn(1, chunk_size * 3, config.hidden_size).to(device)

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
