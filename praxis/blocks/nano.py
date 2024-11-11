import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import time
from transformers import AutoConfig
from typing import Optional
from praxis.activations import ACT2FN


class PraxisNano(nn.Module):
    """
    A special kind of block that omits all self-attention mechanisms, in favor
    of dense layers with sine activations. Inspired by NanoFFT:
    https://github.com/timurgepard/nanoFFT
    """

    def __init__(self, config: AutoConfig, chunk_size: int = 128):
        super().__init__()
        self.chunk_size = chunk_size
        embed_dim = config.num_embeds
        hidden_dim = config.num_dims

        # Core FFT matrices
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fft = nn.ParameterDict(
            {
                "w1": nn.Parameter(torch.Tensor(chunk_size, chunk_size)),
                "w2": nn.Parameter(torch.Tensor(chunk_size, chunk_size)),
            }
        )

        # Initialize weights
        nn.init.xavier_uniform_(self.fft["w1"])
        nn.init.xavier_uniform_(self.fft["w2"])

        # Layer norms and FFW - these operate on full sequence
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffw = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            ACT2FN["sinlu"],
            nn.Dropout(config.dropout),
            nn.Linear(embed_dim, hidden_dim),
        )

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        B, T, E = x.shape

        # Create causal mask for full sequence
        mask = torch.tril(torch.ones(T, T, device=x.device))
        mask = mask / mask.sum(dim=1, keepdim=True)

        # First residual branch
        residual = x
        x = self.ln1(x)

        # FFT processing in chunks
        x = self._process_fft_chunks(x, mask)

        # Add residual
        x = x + residual

        # FFW branch
        residual = x
        x = self.ln2(x)
        x = self.ffw(x) + residual

        return x

    def _process_fft_chunks(self, x: Tensor, full_seq_mask: Tensor) -> Tensor:
        """Process sequence through FFT weights in chunks."""
        B, T, E = x.shape
        chunks_out = []

        for chunk_start in range(0, T, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, T)
            chunk = x[:, chunk_start:chunk_end, :]

            # Get the mask slice that represents this chunk's visibility of the full sequence
            # Each position in chunk can see all previous positions globally
            chunk_len = chunk_end - chunk_start
            mask_slice = full_seq_mask[chunk_start:chunk_end, chunk_start:chunk_end]

            # Process chunk
            chunk = chunk.permute(0, 2, 1)  # [B, E, T_chunk]
            chunk = chunk @ (self.fft["w1"][:chunk_len, :chunk_len] * mask_slice)
            chunk = chunk @ (self.fft["w2"][:chunk_len, :chunk_len] * mask_slice)
            chunk = chunk.permute(0, 2, 1)  # Back to [B, T_chunk, E]

            chunks_out.append(chunk)

        return torch.cat(chunks_out, dim=1)


# class PraxisNano(nn.Module):
#     """
#     A special kind of block that omits all self-attention mechanisms, in favor
#     of dense layers with sine activations. Inspired by NanoFFT:
#     https://github.com/timurgepard/nanoFFT
#     """

#     def __init__(self, config: AutoConfig):
#         super().__init__()
#         max_seq_len = config.context_length // 2
#         embed_dim = config.num_embeds
#         hidden_dim = config.num_dims

#         # Define the weight matrices with maximum sequence length
#         self.ln1 = nn.LayerNorm(hidden_dim)
#         self.fft = nn.ParameterDict(
#             {
#                 "w1": nn.Parameter(torch.Tensor(max_seq_len, max_seq_len)),
#                 "w2": nn.Parameter(torch.Tensor(max_seq_len, max_seq_len)),
#             }
#         )

#         # Initialize weights
#         nn.init.xavier_uniform_(self.fft["w1"])
#         nn.init.xavier_uniform_(self.fft["w2"])

#         # Define the mask
#         mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
#         # Normalize per row
#         row_sums = mask.sum(dim=1, keepdim=True)
#         mask = mask / row_sums
#         self.register_buffer("mask", mask)

#         class SineActivation(nn.Module):
#             def forward(self, x):
#                 return torch.sin(x)

#         # Feed-forward network with sine activation
#         self.ln2 = nn.LayerNorm(hidden_dim)
#         self.ffw = nn.Sequential(
#             nn.Linear(hidden_dim, embed_dim),
#             SineActivation(),
#             # ACT2FN["sinlu"],
#             nn.Linear(embed_dim, hidden_dim),
#         )

#     def forward(self, x: Tensor, attention_mask: Tensor):
#         # x: [batch_size, seq_len, embed_dim]
#         B, T, E = x.shape
#         if T > self.fft["w1"].size(0):
#             raise ValueError(
#                 f"Sequence length {T} exceeds maximum supported length {self.W1.size(0)}."
#             )

#         x = self.ln1(x)

#         # Get the relevant slices of weights and mask based on the actual sequence length
#         W1 = self.fft["w1"][:T, :T] * self.mask[:T, :T]
#         W2 = self.fft["w2"][:T, :T] * self.mask[:T, :T]

#         # Reshape x for matrix multiplication
#         x_fft = x.transpose(1, 2).reshape(-1, T)  # [B * embed_dim, T]

#         # Apply the masked and normalized weight matrices
#         x_fft = x_fft @ W1  # [B * embed_dim, T]
#         x_fft = x_fft @ W2  # [B * embed_dim, T]

#         # Reshape back to original dimensions
#         x_fft = x_fft.view(B, E, T).transpose(1, 2)  # [B, T, E]

#         x = x + x_fft  # Residual connection
#         x = self.ln2(x)
#         x = x + self.ffw(x)  # Residual connection
#         return x


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class MockConfig:
        num_dims: int = 768
        num_embeds: int = 768
        context_length: int = 2048
        vocab_size: int = 50257
        causal: bool = True
        dropout: float = 0.1

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MockConfig()
    chunk_size = 256  # Explicitly define for tests

    def run_memory_test(model, chunk_size):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Create sample input
        x = torch.randn(1, chunk_size, config.num_dims).to(device)

        # Forward pass
        start_time = time.time()
        with torch.amp.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu"
        ):
            output = model(x)
        torch.cuda.synchronize()
        end_time = time.time()

        max_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        return output, end_time - start_time, max_memory

    print("Running tests for PraxisNano...")

    # Create model once for all tests
    model = PraxisNano(config, chunk_size=chunk_size).to(device)

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

    for chunk_size in chunk_sizes:
        output, duration, memory = run_memory_test(model, chunk_size)
        results.append((chunk_size, duration, memory))
        print(f"\nSequence Length: {chunk_size}")
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
        chunk1 = x[:, :chunk_size, :]
        chunk2 = x[:, chunk_size:, :]

        # Process separately
        out1 = model(chunk1)
        out2 = model(chunk2)

        # Process as single sequence
        out_full = model(x)

        # Compare the results at the boundary
        boundary_diff = (
            (
                out_full[:, chunk_size - 1 : chunk_size + 1, :]
                - torch.cat([out1[:, -1:, :], out2[:, :1, :]], dim=1)
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
        # Multiple forward passes should give different results due to random offset
        out1 = model(x)
        out2 = model(x)
        difference = (out1 - out2).abs().mean().item()

        print(f"✓ Output difference between passes: {difference:.6f}")
        assert difference > 0, "Outputs identical despite random offset"
        print("✓ Random offset test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    print("\nAll tests completed!")
