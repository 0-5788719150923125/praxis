"""
FlexAttention module for causal language modeling using PyTorch's FlexAttention.
Based on the efficient attention implementation with block masking support.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FlexAttention(nn.Module):
    """
    Causal self-attention using PyTorch's FlexAttention API.
    Provides efficient attention computation with customizable block masking.
    """

    can_compile = False  # FlexAttention with score_mod triggers FX tracing errors in torch.compile

    def __init__(self, config) -> None:
        """
        Initialize FlexAttention module.

        Args:
            config: Configuration object containing attention parameters
        """
        super().__init__()

        hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_queries = config.num_queries
        self.num_query_heads = self.num_heads * self.num_queries
        self.head_dim = getattr(config, "head_size") or hidden_size // self.num_heads
        self.dropout = config.dropout
        self.causal = config.causal

        # QKV projection - separate sizes for Q (with num_queries) and K/V
        # Q: num_query_heads * head_dim
        # K: num_heads * head_dim
        # V: num_heads * head_dim
        qkv_dim = (
            self.num_query_heads * self.head_dim + 2 * self.num_heads * self.head_dim
        )
        self.qkv = nn.Linear(hidden_size, qkv_dim, bias=False)

        # Output projection
        self.output = nn.Linear(
            self.num_query_heads * self.head_dim, hidden_size, bias=False
        )

        # Dropout layers
        self.dropout = nn.Dropout(self.dropout)

        # Try to import FlexAttention components
        self.flex_attention = None
        self.create_block_mask = None
        self._import_flex_attention()

        # Cache for block masks at different sequence lengths
        # Key is (seq_len, device_str) tuple
        self.block_mask_cache = {}

        # ALiBi slopes for positional encoding
        self.register_buffer(
            "alibi_slopes",
            self._get_alibi_slopes(self.num_query_heads),
            persistent=False,
        )

    def _import_flex_attention(self) -> None:
        """Import FlexAttention components."""
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        self.flex_attention = flex_attention
        self.create_block_mask = create_block_mask

    def _get_alibi_slopes(self, num_heads: int) -> torch.Tensor:
        """
        Compute ALiBi slopes for the given number of heads.

        Args:
            num_heads: Number of attention heads

        Returns:
            Tensor of slopes with shape (num_heads,)
        """
        # Get closest power of 2
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))

        # Get slopes for the closest power of 2
        base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
        powers = torch.arange(1, closest_power_of_2 + 1)
        slopes = torch.pow(base, powers)

        # If num_heads is not a power of 2, interpolate additional slopes
        if closest_power_of_2 != num_heads:
            extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
            num_remaining = num_heads - closest_power_of_2
            extra_powers = torch.arange(1, 2 * num_remaining + 1, 2)
            extra_slopes = torch.pow(extra_base, extra_powers)
            slopes = torch.cat([slopes, extra_slopes], dim=0)

        return slopes[:num_heads]

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create a causal block mask for the given sequence length.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Block mask for causal attention
        """

        # Create cache key using device string representation
        cache_key = (seq_len, str(device))

        # Check cache first
        if cache_key in self.block_mask_cache:
            return self.block_mask_cache[cache_key]

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        # Create block mask (broadcasting over batch and heads)
        block_mask = self.create_block_mask(
            causal_mask,
            B=None,  # Broadcast over batch
            H=None,  # Broadcast over heads
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )

        # Cache the mask with device-specific key
        self.block_mask_cache[cache_key] = block_mask

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
        Forward pass of the FlexAttention module.

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

        # Create or retrieve causal mask
        block_mask = (
            self._create_causal_mask(seq_len, inputs.device) if self.causal else None
        )

        # Create ALiBi score modification function
        alibi_slopes = self.alibi_slopes.to(inputs.device)

        def alibi_score_mod(score, b, h, q_idx, kv_idx):
            """Apply ALiBi positional bias to attention scores."""
            # Calculate position difference (q_idx - kv_idx)
            # This will be negative or zero for causal attention
            bias = alibi_slopes[h] * (kv_idx - q_idx)
            return score + bias

        # Apply FlexAttention with causal mask, ALiBi, and GQA support
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
