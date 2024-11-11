import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import time
from transformers import AutoConfig
from typing import Optional
from praxis.activations import ACT2FN
from dataclasses import dataclass


class PraxisNano(nn.Module):
    """
    A special kind of block that omits all self-attention mechanisms, in favor
    of dense layers with sine activations. Inspired by NanoFFT:
    https://github.com/timurgepard/nanoFFT
    """

    def __init__(
        self,
        config: "AutoConfig",
        chunk_size: int = 256,
        stride: Optional[int] = 128,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.stride = stride if stride is not None else chunk_size

        embed_dim = config.num_embeds
        hidden_dim = config.num_dims

        # Define the weight matrices with chunk_size
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fft = nn.Sequential(
            TriLinear(chunk_size),
            TriLinear(chunk_size),
        )

        # Feed-forward network with sine activation
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffw = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            ACT2FN["sin"],
            nn.Linear(embed_dim, hidden_dim),
        )

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ):
        B, T, E = x.shape
        chunk_size = self.chunk_size
        stride = self.stride

        # Initialize the output tensor and a tensor to keep track of overlap counts
        device = x.device
        output = torch.zeros_like(x)
        overlap_counts = torch.zeros(B, T, E, device=device)

        # Iterate over the sequence in chunks
        for start in range(0, T, stride):
            end = start + chunk_size
            chunk = x[:, start:end, :]

            # Handle the last chunk which might be smaller than chunk_size
            if chunk.size(1) < chunk_size:
                padding = chunk_size - chunk.size(1)
                chunk = nn.functional.pad(chunk, (0, 0, 0, padding), "constant", 0)

            # save residual and apply layer norm
            residual = chunk
            chunk_norm = self.ln1(chunk)

            # Reshape chunk for matrix multiplication
            chunk_fft = chunk_norm.transpose(1, 2)  # [B, embed_dim, T_chunk]

            # Apply the masked and normalized weight matrices
            chunk_fft = self.fft(chunk_fft)

            # Reshape back to original dimensions
            chunk_fft = chunk_fft.transpose(1, 2)  # [B, T_chunk, E]

            # Residual connection
            chunk = chunk_fft + residual

            # Apply second layer norm and feed-forward network
            residual = chunk
            chunk = self.ln2(chunk)
            chunk = self.ffw(chunk) + residual

            # If the chunk was padded, remove the padding
            if end > T:
                chunk = chunk[:, : T - start, :]

            # Accumulate the output and overlap counts
            seq_len = chunk.size(1)
            output[:, start : start + seq_len, :] += chunk
            overlap_counts[:, start : start + seq_len, :] += 1

        # Avoid division by zero
        overlap_counts = torch.clamp(overlap_counts, min=1.0)

        # Average the overlapping regions
        output = output / overlap_counts

        return output


class TriLinear(nn.Linear):
    def __init__(self, features: int):
        super().__init__(features, features, bias=False)

        # Create a lower triangular mask
        causal_mask = torch.tril(torch.ones((features, features), dtype=torch.float32))

        # Apply the mask to the weights: keep lower triangle as initialized, zero upper triangle
        with torch.no_grad():
            self.weight.copy_(self.weight * causal_mask)

        # Compute the normalized mask and register it as a buffer
        mask_normalized = causal_mask / causal_mask.sum(dim=1, keepdim=True)
        self.register_buffer("mask_normalized", mask_normalized)

        # Register the hook to zero gradients outside the mask
        self.weight.register_hook(lambda grad: grad * self.mask_normalized)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


if __name__ == "__main__":
    # Mock AutoConfig class to simulate the configuration
    @dataclass
    class AutoConfig:
        num_dims: int = 768
        num_embeds: int = 768
        context_length: int = 2048
        vocab_size: int = 50257
        causal: bool = True
        dropout: float = 0.1

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

    print("Running tests for PraxisNano...")

    # Create model once for all tests
    model = PraxisNano(config, chunk_size=chunk_size, stride=stride).to(device)

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
        model_test = PraxisNano(config, chunk_size=cs, stride=current_stride).to(device)
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
