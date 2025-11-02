"""
Sliding Window FlexAttention implementation.
Extends the base FlexAttention with configurable sliding window patterns.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from praxis.attention.hex import HexAttention


class SlidingWindowFlexAttention(HexAttention):
    """
    FlexAttention with sliding window support.

    Inherits from FlexAttention and adds sliding window masking capability.
    The sliding window limits attention to a local context window for efficiency.
    """

    def __init__(self, config, window_size: int = 1024) -> None:
        """
        Initialize SlidingWindowFlexAttention module.

        Args:
            config: Configuration object containing attention parameters
            window_size: Size of the sliding attention window
        """
        super().__init__(config)
        self.window_size = window_size

        # Cache for sliding window masks at different sequence lengths
        # Key is (seq_len, window_size, device_str) tuple
        self.sliding_window_cache = {}

    def _create_sliding_window_causal_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create a sliding window + causal block mask for the given sequence length.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Block mask for sliding window + causal attention
        """
        # Create cache key using device string representation
        cache_key = (seq_len, self.window_size, str(device))

        # Check cache first
        if cache_key in self.sliding_window_cache:
            return self.sliding_window_cache[cache_key]

        def sliding_window_causal_mask(b, h, q_idx, kv_idx):
            """Mask function combining causal and sliding window constraints."""
            # Causal constraint: can only attend to previous positions
            causal_mask = q_idx >= kv_idx
            # Window constraint: can only attend within window_size positions
            window_mask = (q_idx - kv_idx) <= self.window_size
            return causal_mask & window_mask

        # Create block mask (broadcasting over batch and heads)
        block_mask = self.create_block_mask(
            sliding_window_causal_mask,
            B=None,  # Broadcast over batch
            H=None,  # Broadcast over heads
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )

        # Cache the mask with device-specific key
        self.sliding_window_cache[cache_key] = block_mask

        return block_mask

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        block_ids: Optional[Tensor] = None,
        current_depth: int = 0,
    ) -> Tuple[Tensor, Optional[Tensor], float]:
        """
        Forward pass of the SlidingWindowFlexAttention module.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask tensor for padding tokens
            past_key_values: Optional cache for key/value pairs (not currently supported)
            block_ids: Optional tensor indicating block structure
            current_depth: Current depth in the network (for caching)

        Returns:
            Tuple containing:
            - Output tensor after attention and projection
            - Updated cache (if using caching)
            - Auxiliary loss value (always 0 for this implementation)
        """
        batch_size, seq_len, _ = inputs.shape

        # Calculate QKV
        qkv = self.qkv(inputs)

        # Split into Q, K, V with proper dimensions
        q_dim = self.num_query_heads * self.head_dim
        kv_dim = self.num_heads * self.head_dim

        q = qkv[..., :q_dim]
        k = qkv[..., q_dim : q_dim + kv_dim]
        v = qkv[..., q_dim + kv_dim :]

        # Reshape for multi-head attention
        # (B, T, num_heads * head_dim) -> (B, num_heads, T, head_dim)
        q = q.view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(
            1, 2
        )

        # For K and V, we always use num_heads (not num_query_heads) for GQA support
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Determine if we're using GQA
        is_gqa = self.num_queries > 1

        # Create sliding window + causal mask
        block_mask = self._create_sliding_window_causal_mask(seq_len, inputs.device)

        # Create ALiBi score modification function
        alibi_slopes = self.alibi_slopes.to(inputs.device)

        def alibi_score_mod(score, b, h, q_idx, kv_idx):
            """Apply ALiBi positional bias to attention scores."""
            # Calculate position difference (q_idx - kv_idx)
            # This will be negative or zero for causal attention
            bias = alibi_slopes[h] * (kv_idx - q_idx)
            return score + bias

        # Apply FlexAttention with sliding window mask, ALiBi, and GQA support
        # Note: scale defaults to 1/sqrt(head_dim) which is what we want
        attn_output = self.flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            score_mod=alibi_score_mod,
            enable_gqa=is_gqa,
            scale=None,  # Use default: 1.0 / sqrt(head_dim)
        )

        # Reshape back: (B, num_heads, T, head_dim) -> (B, T, num_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.output(attn_output)
        output = self.dropout(output)

        # Return output, cache (None for now), and aux_loss (0)
        return output, past_key_values, 0.0
