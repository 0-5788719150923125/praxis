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

    def __init__(self, config: AutoConfig, chunk_size: int = 256):
        super().__init__()
        assert (
            config.causal
        ), "The PraxisNano module was designed for causal language modeling. It wouldn't make sense to use it for other tasks."
        hidden_dim = config.num_dims
        embed_dim = config.num_embeds
        self.chunk_size = chunk_size

        # Core weight matrices
        self.fft_norm = nn.LayerNorm(hidden_dim)
        self.fft = nn.ParameterDict(
            {
                "w1": nn.Parameter(torch.Tensor(self.chunk_size, self.chunk_size)),
                "w2": nn.Parameter(torch.Tensor(self.chunk_size, self.chunk_size)),
            }
        )

        # Initialize weights with triangular structure
        with torch.no_grad():
            nn.init.xavier_uniform_(self.fft["w1"])
            nn.init.xavier_uniform_(self.fft["w2"])
            self.fft["w1"].copy_(torch.tril(self.fft["w1"]))
            self.fft["w2"].copy_(torch.tril(self.fft["w2"]))

        # Create mask for gradient hooks
        mask = torch.tril(torch.ones(self.chunk_size, self.chunk_size))
        row_sums = mask.sum(dim=1, keepdim=True)
        self.register_buffer("base_mask", mask / row_sums)

        # Register gradient hooks
        self.fft["w1"].register_hook(lambda grad: grad * self.base_mask)
        self.fft["w2"].register_hook(lambda grad: grad * self.base_mask)

        # Layer norms and FFN
        self.ffw_norm = nn.LayerNorm(hidden_dim)
        self.ffw = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            ACT2FN["sinlu"],
            nn.Linear(embed_dim, hidden_dim),
        )

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        router_weights: Optional[Tensor] = None,
        token_indices: Optional[Tensor] = None,
    ) -> Tensor:
        B, T, E = x.shape

        residual = x

        x = self.fft_norm(x)

        if T <= self.chunk_size:
            x_fft = self._process_sequence(x)
        else:
            # Generate random initial offset
            offset = torch.randint(0, self.chunk_size, (1,)).item()
            chunks = []

            # Process first chunk
            first_chunk_size = min(self.chunk_size - offset, T)
            if first_chunk_size > 0:
                first_chunk = x[:, :first_chunk_size, :]
                chunk_out = self._process_sequence(
                    first_chunk,
                    pad_left=offset,
                    pad_right=self.chunk_size - offset - first_chunk_size,
                )
                chunks.append(chunk_out)

            # Process middle chunks
            for start_idx in range(
                first_chunk_size, T - self.chunk_size + 1, self.chunk_size
            ):
                chunk = x[:, start_idx : start_idx + self.chunk_size, :]
                chunks.append(self._process_sequence(chunk))

            # Process last chunk if needed
            remaining = T - (
                first_chunk_size
                + ((T - first_chunk_size) // self.chunk_size) * self.chunk_size
            )
            if remaining > 0:
                last_chunk = x[:, T - remaining :, :]
                chunk_out = self._process_sequence(
                    last_chunk, pad_right=self.chunk_size - remaining
                )
                chunks.append(chunk_out)

            x_fft = torch.cat(chunks, dim=1)

        assert x.shape == x_fft.shape, f"Shape mismatch: {x.shape} vs {x_fft.shape}"
        x = x_fft + residual
        residual = x
        x = self.ffw_norm(x)
        x = self.ffw(x) + residual
        return x

    def _process_sequence(
        self, x: Tensor, pad_left: int = 0, pad_right: int = 0
    ) -> Tensor:
        """Process a sequence with optional padding."""
        B, T, E = x.shape

        # Handle padding
        if pad_left > 0 or pad_right > 0:
            x = F.pad(x, (0, 0, pad_left, pad_right))

        # Process through FFT in BTE format
        x = x.permute(0, 2, 1)  # BET -> BTE
        x = x @ self.fft["w1"][: x.size(2), : x.size(2)]
        x = x @ self.fft["w2"][: x.size(2), : x.size(2)]
        x = x.permute(0, 2, 1)  # BTE -> BET

        # Remove padding
        if pad_left > 0:
            x = x[:, pad_left:, :]
        if pad_right > 0:
            x = x[:, :-pad_right, :]

        return x


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class MockConfig:
        num_dims: int = 768
        num_embeds: int = 768
        context_length: int = 2048
        vocab_size: int = 50257
        causal: bool = True

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
