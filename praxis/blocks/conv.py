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
    inspired by NanoFFT, this module looks almost nothing like that now.
    https://github.com/timurgepard/nanoFFT
    Periodic activation functions can train a model to "know what they do not know."
    https://arxiv.org/abs/2110.13572
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig", *args, **kwargs):
        super().__init__()
        hidden_dim = config.num_dims

        reduction = 0.75
        self.fft_norm = nn.LayerNorm(hidden_dim)
        self.fft = CausalPeriodicConvolution(config, reduction=reduction)

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
        # Normalize input
        x_norm = self.fft_norm(x)  # (B, T, E)

        # Transpose to (B, E, T) for Conv1d
        x_out = x_norm.transpose(1, 2)  # (B, E, T)

        # Capture local and global patterns
        x_out = self.fft(x_out)

        # Transpose back to (B, T, E)
        x_out = x_out.transpose(1, 2)  # (B, T, reduced_dim)

        # Residual connection
        residual = x_out + x

        # Feedforward network
        x_norm = self.ffw_norm(residual)
        x_ffw = self.ffw(x_norm)
        return x_ffw + residual


class CausalPeriodicConvolution(nn.Module):
    def __init__(self, config, reduction, *args, **kwargs):
        super().__init__()
        hidden_dim = config.num_dims

        # First Causal Convolution
        self.conv1 = CausalConv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            dilation=1,
        )
        self.dropout = nn.Dropout(config.dropout)

        # Split Dimension
        split_dim = hidden_dim // 2

        # Second Causal Convolution Path
        self.conv2 = CausalConv1d(
            in_channels=split_dim,
            out_channels=split_dim,
            kernel_size=3,
            dilation=2,
        )

        # Global Context Path
        self.conv3 = CausalGlobalContext(split_dim, reduction)

        # Blend the features
        self.output = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, x: Tensor):
        identity = x  # Save input for residual connection

        # First Causal Convolution
        x_conv = self.conv1(x)
        x_conv = self.dropout(x_conv)

        # Split into two parts
        x_local, x_global = torch.chunk(x_conv, 2, dim=1)

        # Process local path
        x_local = self.conv2(x_local)

        # Process global path
        x_global = self.conv3(x_global)

        # Concatenate along the channel dimension
        x_concat = torch.cat([x_local, x_global], dim=1)

        # Final output projection with residual connection
        return self.output(x_concat) + identity


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

    def __init__(self, in_channels, reduction=0.125):
        super().__init__()
        bottleneck = int(in_channels * reduction)

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

    def forward(self, x):
        # Generate attention weights
        attn = self.context(x)  # B, 1, T

        # Apply causal masking
        mask = torch.triu(torch.ones_like(attn), diagonal=1)
        attn = attn.masked_fill(mask.bool(), float("-inf"))
        attn = F.softmax(attn, dim=-1)  # B, 1, T

        # Calculate global context
        context = torch.matmul(x, attn.transpose(-2, -1))  # B, C, 1

        # Transform through bottleneck (no need to squeeze/unsqueeze)
        context = self.transform(context)  # B, C, 1

        # Broadcast and add to input
        return x + context.expand(-1, -1, x.size(2))


# class CausalGlobalContext(nn.Module):
#     """
#     Implements a kind of squeeze-and-excitation mechanism, which allows
#     us to bridge convolutional operations' local contexts, into a global one.
#     https://arxiv.org/abs/1904.11492v1
#     """

#     def __init__(self, in_channels, reduction=0.125):
#         super().__init__()
#         bottleneck = int(in_channels * reduction)

#         # Context modeling - single 1x1 conv to generate global attention weights
#         self.context = nn.Conv1d(in_channels, 1, kernel_size=1)

#         # Bottleneck transform
#         self.transform = nn.Sequential(
#             nn.Linear(in_channels, bottleneck),
#             # nn.Conv1d(in_channels, 1, kernel_size=1),
#             nn.LayerNorm(bottleneck),
#             ACT2FN["periodic_relu"],
#             # nn.Conv1d(1, in_channels, kernel_size=1),
#             nn.Linear(bottleneck, in_channels),
#         )

#     def forward(self, x):
#         # Generate attention weights
#         attn = self.context(x)  # B, 1, T

#         # Apply causal masking by setting future weights to -inf
#         mask = torch.triu(torch.ones_like(attn), diagonal=1)
#         attn = attn.masked_fill(mask.bool(), float("-inf"))

#         # Softmax to get attention distribution
#         attn = F.softmax(attn, dim=-1)  # B, 1, T

#         # Calculate global context
#         context = torch.matmul(x, attn.transpose(-2, -1))  # B, C, 1
#         context = context.squeeze(-1)  # B, C

#         # Transform through bottleneck
#         context = self.transform(context)  # B, C

#         # Add to all positions
#         return x + context.unsqueeze(-1)  # B, C, T


# class CausalGlobalContext(nn.Module):
#     def __init__(self, in_channels, reduction=0.125):
#         super().__init__()
#         bottleneck = int(in_channels * reduction)

#         # Context modeling with causal masking
#         self.query = nn.Conv1d(in_channels, 1, kernel_size=1)
#         self.key = nn.Conv1d(in_channels, 1, kernel_size=1)
#         self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)

#         self.transform = nn.Sequential(
#             nn.Linear(in_channels, bottleneck),
#             nn.LayerNorm(bottleneck),
#             ACT2FN["periodic_relu"],
#             nn.Linear(bottleneck, in_channels),
#         )

#     def forward(self, x):
#         B, C, T = x.shape

#         # Generate causal attention mask
#         mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
#         mask = mask.to(x.device)

#         # Compute query-key attention scores
#         q = self.query(x).transpose(1, 2)  # B, T, 1
#         k = self.key(x).transpose(1, 2)  # B, T, 1
#         v = self.value(x)  # B, C, T

#         # Compute attention weights with causality
#         attn = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(1)  # B, T, T
#         attn = attn.masked_fill(mask, float("-inf"))
#         attn = F.softmax(attn, dim=-1)  # B, T, T

#         # Compute global context per position
#         context = torch.bmm(v, attn.transpose(-2, -1))  # B, C, T

#         # Transform and fuse
#         context = context.transpose(1, 2)  # B, T, C
#         context = self.transform(context)  # B, T, C
#         context = context.transpose(1, 2)  # B, C, T

#         return x + context


# class CausalGlobalContext(nn.Module):
#     """
#     Implements a kind of squeeze-and-excitation mechanism, which allows
#     us to bridge convolutional operations' local contexts, into a global one.
#     https://arxiv.org/abs/1904.11492v1
#     """

#     def __init__(self, in_channels, reduction=0.125):
#         super().__init__()

#         bottleneck = int(in_channels * reduction)

#         # Context Modeling: 1x1 convolution to generate attention scores
#         self.context = nn.Conv1d(in_channels, 1, kernel_size=1)

#         # Transform Module: Bottleneck with LayerNorm and ReLU
#         self.transform = nn.Sequential(
#             nn.Linear(in_channels, bottleneck, bias=False),
#             nn.LayerNorm(bottleneck),
#             ACT2FN["periodic_relu"],
#             nn.Linear(bottleneck, in_channels, bias=False),
#         )

#     def forward(self, x):
#         # Compute attention scores
#         attn_scores = self.context(x).squeeze(1)  # Shape: (B, T)
#         attn_weights = F.softmax(attn_scores, dim=1)  # Shape: (B, T)

#         # Compute cumulative attention weights for causality
#         cumulative_attn_weights = torch.cumsum(attn_weights, dim=1)  # Shape: (B, T)

#         # Normalize cumulative attention weights
#         cumulative_attn_weights = cumulative_attn_weights / (
#             cumulative_attn_weights[:, -1].unsqueeze(1) + 1e-6
#         )  # Shape: (B, T)

#         # **Add an extra dimension to attn_weights for broadcasting**
#         attn_weights = attn_weights.unsqueeze(1)  # Shape: (B, 1, T)

#         # Compute cumulative sum of input features weighted by attention weights
#         weighted_input = x * attn_weights  # Shape: (B, C, T)
#         cumulative_context = torch.cumsum(weighted_input, dim=2)  # Shape: (B, C, T)

#         # **Add an extra dimension to cumulative_attn_weights for broadcasting**
#         cumulative_attn_weights = cumulative_attn_weights.unsqueeze(
#             1
#         )  # Shape: (B, 1, T)

#         # Normalize cumulative context
#         context = cumulative_context / (
#             cumulative_attn_weights + 1e-6
#         )  # Shape: (B, C, T)

#         # Use the last time step context for each position
#         context = context[:, :, -1]  # Shape: (B, C)

#         # Transform Module
#         transformed = self.transform(context)  # Shape: (B, C)

#         # Fusion: Add transformed context to each position
#         out = x + transformed.unsqueeze(2)  # Shape: (B, C, T)

#         return out


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
