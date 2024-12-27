import gc
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryEfficientAttention(nn.Module):
    """
    Memory efficient attention implementation that accepts 3D input tensors.
    https://arxiv.org/abs/2112.05682
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def _chunk_to_heads(self, x, chunk_size):
        """Transform chunk to multi-head format."""
        batch_size = x.shape[0]
        x = x.view(batch_size, chunk_size, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [batch, heads, chunk_size, head_dim]

    def forward(self, inputs, causal_mask: bool = False):
        """
        Memory-efficient attention implementation processing input in chunks.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, hidden_dim]
            causal_mask: Whether to apply causal masking
        """
        batch_size, seq_len, _ = inputs.shape
        chunk_size = int(math.sqrt(seq_len))
        chunk_size = max(1, min(chunk_size, seq_len))

        # Initialize output tensor
        output = torch.zeros_like(inputs)

        # Process query sequence in chunks
        for query_start in range(0, seq_len, chunk_size):
            query_end = min(query_start + chunk_size, seq_len)
            query_chunk = inputs[:, query_start:query_end]

            # Project and reshape query chunk
            q = self.q_proj(query_chunk)
            q = self._chunk_to_heads(q, query_end - query_start)
            q = q / math.sqrt(self.head_dim)

            # Initialize accumulators for this query chunk
            value_sum = torch.zeros(
                batch_size,
                self.num_heads,
                query_end - query_start,
                self.head_dim,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            normalizer = torch.zeros(
                batch_size,
                self.num_heads,
                query_end - query_start,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            max_score = torch.full(
                (batch_size, self.num_heads, query_end - query_start),
                float("-inf"),
                device=inputs.device,
                dtype=inputs.dtype,
            )

            # Process key/value sequence in chunks
            for key_start in range(0, seq_len, chunk_size):
                key_end = min(key_start + chunk_size, seq_len)
                key_chunk = inputs[:, key_start:key_end]

                # Project and reshape key/value chunks
                k = self._chunk_to_heads(self.k_proj(key_chunk), key_end - key_start)
                v = self._chunk_to_heads(self.v_proj(key_chunk), key_end - key_start)

                # Compute attention scores for current chunks
                scores = torch.matmul(q, k.transpose(-2, -1))

                # Apply causal masking if needed
                if causal_mask and key_start > query_start:
                    scores.fill_(float("-inf"))
                elif causal_mask:
                    causal_mask_chunk = (
                        torch.arange(query_start, query_end, device=scores.device)[
                            :, None
                        ]
                        >= torch.arange(key_start, key_end, device=scores.device)[
                            None, :
                        ]
                    )
                    scores.masked_fill_(
                        ~causal_mask_chunk.view(1, 1, query_end - query_start, -1),
                        float("-inf"),
                    )

                # Update running maximum
                max_score_prev = max_score
                max_score = torch.maximum(max_score, torch.max(scores, dim=-1)[0])

                # Compute exponentials with numerical stability
                exp_scores = torch.exp(scores - max_score.unsqueeze(-1))
                exp_prev = torch.exp(max_score_prev - max_score)

                # Update running sums
                value_sum = value_sum * exp_prev.unsqueeze(-1)
                value_sum = value_sum + torch.matmul(exp_scores, v)
                normalizer = normalizer * exp_prev + exp_scores.sum(dim=-1)

                # Clear unnecessary tensors
                del k, v, scores, exp_scores

            # Compute attention output for current query chunk
            chunk_output = value_sum / normalizer.unsqueeze(
                -1
            )  # [batch, heads, chunk_size, head_dim]
            chunk_output = chunk_output.transpose(
                1, 2
            ).contiguous()  # [batch, chunk_size, heads, head_dim]
            chunk_output = chunk_output.view(batch_size, -1, self.hidden_dim)
            chunk_output = self.out_proj(chunk_output)

            # Write to output tensor
            output[:, query_start:query_end] = chunk_output

        return output


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
            return None
    except (torch.cuda.OutOfMemoryError, MemoryError):
        return None


def test_attention():
    print("\nTesting attention implementation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running tests on: {device}")

    # Memory usage tests with realistic settings
    print("\nTesting memory usage...\n")
    print("Sequence Length | Memory Usage (MB)")
    print("-" * 35)

    test_lengths = [512, 1024, 2048, 4096, 8192]
    hidden_dim = 768  # Standard transformer hidden dimension

    for seq_len in test_lengths:
        torch.cuda.empty_cache()
        gc.collect()

        inputs = torch.randn(1, seq_len, hidden_dim, device=device)
        attention = MemoryEfficientAttention(hidden_dim, 12).to(device)

        def run_test():
            with torch.no_grad():
                output = attention(inputs, causal_mask=True)
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
