import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import DynamicCache

from praxis.encoding import ENCODING_REGISTRY


class SyntaxesAttention(nn.Module):
    """
    Syntaxes Attention: Global query compression with sliding window K/V.

    This attention mechanism uses asymmetric processing:
    - Queries: Compressed globally to capture long-range dependencies
    - Keys/Values: Limited to recent tokens via sliding window

    All compressed queries attend to the same recent window of tokens,
    maintaining global context while focusing on local details.

    Key features:
    - Global query compression preserves long-range patterns
    - Sliding window K/V reduces memory and focuses on recent context
    - Reduces complexity from O(n²) to O(n/r * w) where r is compression ratio, w is window size
    - Maintains causality through windowed attention
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        # Ensure num_heads divides hidden_size evenly
        while self.hidden_size % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1

        self.head_dim = self.hidden_size // self.num_heads

        # Query compression ratio (how much to reduce query length)
        self.query_compression_ratio = getattr(
            config, "syntaxes_query_compression_ratio", 4
        )

        # Sliding window size for keys/values
        self.window_size = getattr(config, "syntaxes_window_size", 128)

        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Dropout for regularization
        self.dropout = nn.Dropout(getattr(config, "dropout", 0.1))

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Create a modified config for the encoding that reflects the actual number of heads
        encoding_config = type(config)(**vars(config))
        encoding_config.num_heads = self.num_heads
        # Keep num_queries as 1 to maintain the same total query heads
        encoding_config.num_queries = 1

        # Positional encoding
        self.encoding = ENCODING_REGISTRY[config.encoding](encoding_config)

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Union[Tensor, DynamicCache]] = None,
        block_ids: Optional[Tensor] = None,
        current_depth: int = 0,
        offset: int = 0,
    ) -> Tuple[Tensor, Optional[Union[Tensor, DynamicCache]], Union[int, float]]:
        """Forward pass of Syntaxes attention with compressed queries and positional encoding."""
        batch_size, seq_len, hidden_size = inputs.shape
        device = inputs.device

        if past_key_values is not None:
            raise NotImplementedError(
                "KV caching not yet supported for SyntaxesAttention"
            )

        # Project queries from inputs (before compression)
        q = self.q_proj(inputs)  # [batch, seq_len, hidden]

        # Reshape queries for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch, heads, seq_len, head_dim]

        # Apply positional encoding to queries before compression
        # Create dummy k, v with same shape for the encoding API
        k_dummy = torch.zeros_like(q)
        v_dummy = torch.zeros_like(q)
        q, _, _ = self.encoding.before_scores(
            q, k_dummy, v_dummy, offset=offset, block_ids=block_ids
        )

        # Calculate compressed sequence length - we need to round up to handle remainder tokens
        compressed_seq_len = (
            seq_len + self.query_compression_ratio - 1
        ) // self.query_compression_ratio

        # Causal pooling for query compression
        # Use average pooling with causal chunks to maintain temporal order
        if compressed_seq_len < seq_len:
            # Pad sequence to make it evenly divisible by compression ratio
            target_seq_len = compressed_seq_len * self.query_compression_ratio
            pad_len = target_seq_len - seq_len

            if pad_len > 0:
                q = F.pad(q, (0, 0, 0, pad_len), mode="constant", value=0)
                padded_seq_len = target_seq_len
            else:
                padded_seq_len = seq_len

            # Reshape for pooling: [batch, heads, compressed_seq_len, compression_ratio, head_dim]
            q_reshaped = q[:, :, :padded_seq_len, :].view(
                batch_size,
                self.num_heads,
                compressed_seq_len,
                self.query_compression_ratio,
                self.head_dim,
            )

            # Average pool along the compression dimension
            # For padded positions, we need to adjust the mean to account for zeros
            q_compressed = q_reshaped.mean(
                dim=3
            )  # [batch, heads, compressed_seq_len, head_dim]
            if pad_len > 0:
                # Adjust the last compressed position to account for padding
                last_group_size = seq_len % self.query_compression_ratio
                if last_group_size > 0:
                    # Rescale the last position to account for zeros in padding
                    q_compressed[:, :, -1, :] = (
                        q_compressed[:, :, -1, :]
                        * self.query_compression_ratio
                        / last_group_size
                    )
        else:
            # No compression needed
            q_compressed = q

        # Sliding window for keys and values
        # All compressed queries will attend to the same recent window
        window_size = min(self.window_size, seq_len)
        window_start = max(0, seq_len - window_size)

        # Extract the window of recent tokens for K/V
        window_inputs = inputs[:, window_start:, :]  # [batch, window_size, hidden]

        # Project K/V from the window
        k = self.k_proj(window_inputs)  # [batch, window_size, hidden]
        v = self.v_proj(window_inputs)  # [batch, window_size, hidden]

        # Reshape K/V for multi-head attention
        k = k.view(batch_size, window_size, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch, heads, window_size, head_dim]
        v = v.view(batch_size, window_size, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch, heads, window_size, head_dim]

        # Apply positional encoding to K/V with window offset
        # The window starts at position window_start, so we add that to the offset
        window_offset = offset + window_start
        k, v, _ = self.encoding.before_scores(
            k, k, v, offset=window_offset, block_ids=block_ids
        )

        # Attention computation - all compressed queries attend to the same window
        # q_compressed: [batch, heads, compressed_seq_len, head_dim]
        # k: [batch, heads, window_size, head_dim]
        # v: [batch, heads, window_size, head_dim]

        # Compute attention scores
        scores = (
            torch.matmul(
                q_compressed,  # [batch, heads, compressed_seq_len, head_dim]
                k.transpose(-2, -1),  # [batch, heads, head_dim, window_size]
            )
            * self.scale
        )  # [batch, heads, compressed_seq_len, window_size]

        # Apply positional encoding to scores
        scores = self.encoding.after_scores(scores, offset=offset, block_ids=block_ids)

        # Create causal mask
        # Each compressed query position maps to a position in the original sequence
        # and can only attend to K/V positions that come before or at that position
        compressed_positions = torch.arange(compressed_seq_len, device=device)
        # Map compressed positions back to original sequence positions
        original_query_positions = (
            (compressed_positions + 1) * self.query_compression_ratio - 1
        ).unsqueeze(1)
        # Window positions in the original sequence
        window_positions = torch.arange(
            window_start, window_start + window_size, device=device
        ).unsqueeze(0)
        # Causal mask: can only attend to positions <= original query position
        causal_mask = (
            window_positions <= original_query_positions
        )  # [compressed_seq_len, window_size]

        # Special handling: if a query position is before the window, it shouldn't attend to anything
        # But to avoid NaN, we'll let it attend to the first position in the window
        queries_before_window = original_query_positions.squeeze(1) < window_start
        if queries_before_window.any():
            # For queries before window, allow attention to first window position only
            causal_mask[queries_before_window, 0] = True

        causal_mask = causal_mask.unsqueeze(0).unsqueeze(
            0
        )  # [1, 1, compressed_seq_len, window_size]
        scores = scores.masked_fill(~causal_mask, -torch.inf)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Extract window portion of attention mask
            window_mask = attention_mask[:, window_start : window_start + window_size]
            window_mask = window_mask.unsqueeze(1).unsqueeze(
                1
            )  # [batch, 1, 1, window_size]
            scores = scores.masked_fill(window_mask == 0, -torch.inf)

        # Softmax and dropout
        weights = F.softmax(
            scores, dim=-1
        )  # [batch, heads, compressed_seq_len, window_size]
        weights = self.dropout(weights)

        # Apply attention to values
        output = torch.matmul(
            weights,  # [batch, heads, compressed_seq_len, window_size]
            v,  # [batch, heads, window_size, head_dim]
        )  # [batch, heads, compressed_seq_len, head_dim]

        # Reshape and prepare for upsampling
        output = output.transpose(
            1, 2
        ).contiguous()  # [batch, compressed_seq_len, heads, head_dim]
        output = output.view(batch_size, compressed_seq_len, self.hidden_size)

        # Apply output projection before upsampling (more efficient)
        output = self.o_proj(output)  # [batch, compressed_seq_len, hidden]

        # Upsample back to original sequence length using linear interpolation
        # This spreads the compressed attention output back to full resolution
        if compressed_seq_len != seq_len:
            output = output.transpose(1, 2)  # [batch, hidden, compressed_seq_len]
            output = F.interpolate(
                output, size=seq_len, mode="linear", align_corners=False
            )  # [batch, hidden, seq_len]
            output = output.transpose(1, 2)  # [batch, seq_len, hidden]

        # Final dropout
        output = self.dropout(output)

        return output, None, 0
