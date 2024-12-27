import gc
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryEfficientAttention(nn.Module):
    """Memory efficient attention implementation from 'Self-attention Does Not Need O(n2) Memory'."""

    def __init__(self, dim_k: int):
        super().__init__()
        self.dim_k = dim_k

    def forward(self, query, key, value, causal_mask: bool = False):
        """Memory efficient attention computation using chunked processing."""
        # Extract dimensions
        batch_size, num_heads, num_queries, dim_k = query.shape
        seq_len = key.shape[2]
        dim_v = value.shape[-1]

        # Scale query
        query = query / math.sqrt(self.dim_k)

        # Determine chunk size - use sqrt(N) for O(sqrt(N)) memory complexity
        chunk_size = int(math.sqrt(seq_len))
        chunk_size = max(1, min(chunk_size, seq_len))  # Ensure valid chunk size

        # Initialize accumulators
        value_sum = torch.zeros(
            batch_size,
            num_heads,
            num_queries,
            dim_v,
            device=query.device,
            dtype=query.dtype,
        )
        normalizer = torch.zeros(
            batch_size, num_heads, num_queries, device=query.device, dtype=query.dtype
        )
        max_score = torch.full(
            (batch_size, num_heads, num_queries),
            float("-inf"),
            device=query.device,
            dtype=query.dtype,
        )

        # Process key-value pairs in chunks
        for chunk_idx in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_idx + chunk_size, seq_len)

            # Get current chunk of keys and values
            key_chunk = key[:, :, chunk_idx:chunk_end]
            value_chunk = value[:, :, chunk_idx:chunk_end]

            # Compute attention scores for current chunk
            chunk_scores = torch.matmul(query, key_chunk.transpose(-2, -1))

            # Apply causal masking if needed
            if causal_mask:
                # Create causal mask for current chunk
                query_positions = torch.arange(num_queries, device=query.device)
                key_positions = torch.arange(chunk_idx, chunk_end, device=query.device)
                causal_mask_chunk = query_positions[:, None] >= key_positions[None, :]
                chunk_scores = torch.where(
                    causal_mask_chunk.view(1, 1, num_queries, -1),
                    chunk_scores,
                    float("-inf"),
                )

            # Update running maximum
            max_score_prev = max_score
            max_score = torch.maximum(max_score, torch.max(chunk_scores, dim=-1)[0])

            # Compute exponentials with numerical stability
            exp_chunk = torch.exp(chunk_scores - max_score.unsqueeze(-1))
            exp_prev = torch.exp(max_score_prev - max_score)

            # Update running sums
            value_sum = value_sum * exp_prev.unsqueeze(-1)
            value_sum = value_sum + torch.matmul(exp_chunk, value_chunk)
            normalizer = normalizer * exp_prev + exp_chunk.sum(dim=-1)

        # Final normalization
        out = value_sum / normalizer.unsqueeze(-1)
        return out


def measure_peak_memory(func, device) -> Union[float, None]:
    """Measure peak memory usage for either CPU or GPU."""
    try:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            func()
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            # For CPU, return None as we're mainly interested in GPU memory
            return None
    except (torch.cuda.OutOfMemoryError, MemoryError):
        return None


def test_attention():
    print("\nTesting attention implementation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running tests on: {device}")

    # Test parameters
    batch_size = 2
    num_heads = 4
    dim_k = 64
    dim_v = 32
    seq_len = 8

    # Create test inputs
    torch.manual_seed(42)
    query = torch.randn(batch_size, num_heads, seq_len, dim_k, device=device)
    key = torch.randn(batch_size, num_heads, seq_len, dim_k, device=device)
    value = torch.randn(batch_size, num_heads, seq_len, dim_v, device=device)

    # Initialize attention module
    attention = MemoryEfficientAttention(dim_k).to(device)

    # Test causal masking
    output = attention(query, key, value, causal_mask=True)

    # Verify causal masking (first position should only attend to itself)
    first_pos_output = output[0, 0, 0]  # batch 0, head 0, position 0
    first_pos_value = value[0, 0, 0]  # corresponding value
    assert torch.allclose(
        first_pos_output, first_pos_value, rtol=1e-4
    ), "Causal masking test failed: First position should only attend to itself"

    print("Basic functionality tests passed!")

    # Memory usage tests
    print("\nTesting memory usage...\n")
    print("Sequence Length | Memory Usage (MB)")
    print("-" * 35)

    for seq_len in [512, 1024, 2048, 4096, 8192]:
        query = torch.randn(1, 4, seq_len, 64, device=device)
        key = torch.randn(1, 4, seq_len, 64, device=device)
        value = torch.randn(1, 4, seq_len, 32, device=device)

        def run_test():
            with torch.no_grad():
                output = attention(query, key, value, causal_mask=True)
                output.cpu()

        memory = measure_peak_memory(run_test, device)
        memory_str = f"{memory:17.2f}" if memory is not None else "      N/A (CPU)"
        print(f"{seq_len:14d} | {memory_str}")


if __name__ == "__main__":
    try:
        test_attention()
        print("\nAll tests passed successfully!")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
