"""
HexAttention module for causal language modeling using PyTorch's FlexAttention.
Based on the efficient attention implementation with block masking support.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HexAttention(nn.Module):
    """
    Causal self-attention using PyTorch's FlexAttention API.
    Provides efficient attention computation with customizable block masking.
    """

    def __init__(self, config, pos_type: str = None) -> None:
        """
        Initialize HexAttention module.

        Args:
            config: Configuration object containing attention parameters
            pos_type: Positional encoding type - "alibi" or "rope"
                     If None, reads from config.pos_type (default "alibi")
        """
        super().__init__()

        # Get pos_type from parameter or config
        if pos_type is None:
            pos_type = getattr(config, "pos_type", "alibi")

        if pos_type not in ["alibi", "rope"]:
            raise ValueError(f"pos_type must be 'alibi' or 'rope', got {pos_type}")

        self.pos_type = pos_type
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

        # ALiBi slopes (only if using ALiBi)
        if self.pos_type == "alibi":
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

    def _apply_rope(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply Rotary Position Embedding to queries and keys.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]

        Returns:
            Rotated (q, k) tensors
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Generate position indices
        pos = torch.arange(seq_len, device=q.device, dtype=q.dtype)

        # Compute frequency bands: θ_i = 10000^(-2i/d)
        dim_indices = torch.arange(0, head_dim, 2, device=q.device, dtype=q.dtype)
        freqs = 1.0 / (10000.0 ** (dim_indices / head_dim))

        # Outer product: [seq_len, head_dim/2]
        angles = torch.outer(pos, freqs)

        # Compute cos and sin
        cos = angles.cos()
        sin = angles.sin()

        # Reshape for broadcasting: [1, 1, seq_len, head_dim/2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Split q and k into even/odd indices
        q_even = q[..., 0::2]
        q_odd = q[..., 1::2]
        k_even = k[..., 0::2]
        k_odd = k[..., 1::2]

        # Apply rotation
        q_rotated = torch.stack(
            [q_even * cos - q_odd * sin, q_odd * cos + q_even * sin], dim=-1
        ).flatten(-2)

        k_rotated = torch.stack(
            [k_even * cos - k_odd * sin, k_odd * cos + k_even * sin], dim=-1
        ).flatten(-2)

        return q_rotated, k_rotated

    def _create_causal_mask(
        self, q_len: int, kv_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create a causal block mask for the given sequence lengths.

        Args:
            q_len: Query sequence length
            kv_len: Key/Value sequence length (may include ghost token)
            device: Device to create mask on

        Returns:
            Block mask for causal attention with ghostmax support
        """

        # Create cache key using device string representation
        cache_key = (q_len, kv_len, str(device))

        # Check cache first
        if cache_key in self.block_mask_cache:
            return self.block_mask_cache[cache_key]

        def causal_mask(b, h, q_idx, kv_idx):
            # Standard causal: allow attending to positions <= q_idx
            # Ghost token (if present at kv_idx >= q_len) is always accessible
            return (q_idx >= kv_idx) | (kv_idx >= q_len)

        # Create block mask (broadcasting over batch and heads)
        block_mask = self.create_block_mask(
            causal_mask,
            B=None,  # Broadcast over batch
            H=None,  # Broadcast over heads
            Q_LEN=q_len,
            KV_LEN=kv_len,
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

        # Apply positional encoding BEFORE ghostmax (RoPE needs matching seq_len)
        if self.pos_type == "rope":
            q, k = self._apply_rope(q, k)
            score_mod = None
        else:  # alibi
            alibi_bias = self.alibi_slopes.to(inputs.device)
            seq_len_tensor = torch.tensor(
                seq_len, device=inputs.device, dtype=torch.int32
            )

            def alibi_score_mod(score, b, h, q_idx, kv_idx):
                """Apply ALiBi positional bias to attention scores."""
                bias = alibi_bias[h] * (kv_idx - q_idx) * (kv_idx < seq_len_tensor)
                return score + bias

            score_mod = alibi_score_mod

        # Ghostmax: Append zero token to K and V (after positional encoding)
        zero_k = torch.zeros(
            batch_size, self.num_heads, 1, self.head_dim, device=k.device, dtype=k.dtype
        )
        zero_v = torch.zeros(
            batch_size, self.num_heads, 1, self.head_dim, device=v.device, dtype=v.dtype
        )
        k = torch.cat([k, zero_k], dim=2)  # (B, H, T+1, D)
        v = torch.cat([v, zero_v], dim=2)  # (B, H, T+1, D)
        kv_len = seq_len + 1

        # Determine if we're using GQA
        is_gqa = self.num_queries > 1

        # Create or retrieve causal mask (with ghost token support)
        block_mask = (
            self._create_causal_mask(seq_len, kv_len, inputs.device)
            if self.causal
            else None
        )

        # Apply FlexAttention with causal mask and GQA support
        attn_output = self.flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            score_mod=score_mod,
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
