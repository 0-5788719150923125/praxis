"""
CausalAttention module for causal language modeling using PyTorch's FlexAttention.
Based on the efficient attention implementation with block masking support.
"""

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Suppress verbose compile-time logs
os.environ["TORCHDYNAMO_EXTENDED_ADVICE"] = "0"


class CausalAttention(nn.Module):
    """
    Causal self-attention using PyTorch's FlexAttention API.
    Provides efficient attention computation with customizable block masking.
    Supports optional sliding window for efficient long-sequence inference.
    """

    def __init__(self, config) -> None:
        """
        Initialize CausalAttention module.

        Args:
            config: Configuration object containing attention parameters
                   (config.encoding should be "alibi" or "rope")
                   Optional config.window_size (int) enables sliding window attention.
        """
        super().__init__()

        # Get positional encoding type from config
        pos_type = config.encoding

        if pos_type not in ["alibi", "rope"]:
            raise ValueError(
                f"config.encoding must be 'alibi' or 'rope' for CausalAttention, got '{pos_type}'"
            )

        self.pos_type = pos_type
        hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_queries = config.num_queries
        self.num_query_heads = self.num_heads * self.num_queries
        self.head_dim = getattr(config, "head_size") or hidden_size // self.num_heads

        if pos_type == "rope" and self.head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires an even head_dim, got {self.head_dim} "
                f"(hidden_size={hidden_size}, num_heads={self.num_heads})"
            )
        self.dropout_p = config.dropout
        self.causal = config.causal
        self.window_size = getattr(config, "window_size", None)

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
        self.dropout = nn.Dropout(self.dropout_p)

        # Try to import FlexAttention components
        self.flex_attention = None
        self.create_block_mask = None
        self.and_masks = None
        self._import_flex_attention()

        # Cache for block masks at different sequence lengths
        # Key is (seq_len, kv_len, device_str) tuple
        self.block_mask_cache = {}

        # ALiBi slopes (only if using ALiBi)
        if self.pos_type == "alibi":
            self.register_buffer(
                "alibi_slopes",
                self._get_alibi_slopes(self.num_query_heads),
                persistent=False,
            )

    def _import_flex_attention(self) -> None:
        """Import FlexAttention components (GPU only - falls back to SDPA on CPU)."""
        try:
            from torch.nn.attention.flex_attention import (
                and_masks,
                create_block_mask,
                flex_attention,
            )

            self.flex_attention = flex_attention
            self.create_block_mask = create_block_mask
            self.and_masks = and_masks
        except ImportError:
            print("[CausalAttention] FlexAttention not available, using SDPA fallback")
            self.flex_attention = None
            self.create_block_mask = None
            self.and_masks = None

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

    def _build_mask_mod(self):
        """Build a mask_mod closure following the FlexAttention pattern.

        Captures primitive values (not self) so the resulting function is
        compatible with torch.compile tracing inside create_block_mask.
        """
        window_size = self.window_size  # capture as plain int or None

        def ghost_causal_mask(b, h, q_idx, kv_idx):
            # Ghost token (kv_idx=0) is always accessible.
            # Actual tokens: kv_idx=1 is position 0, so q_idx + 1 >= kv_idx.
            return (kv_idx == 0) | (q_idx + 1 >= kv_idx)

        if window_size is None:
            return ghost_causal_mask

        def ghost_sliding_window_mask(b, h, q_idx, kv_idx):
            # Ghost token must also pass here, otherwise and_masks kills it.
            # Actual tokens: limit to window_size positions (ghost-shifted).
            return (kv_idx == 0) | (q_idx - (kv_idx - 1) <= window_size)

        return self.and_masks(ghost_causal_mask, ghost_sliding_window_mask)

    def _create_causal_mask(
        self, q_len: int, kv_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create a block mask with ghostmax support, optional sliding window.

        Uses composable mask_mod functions per the FlexAttention API:
        causal and sliding window constraints are composed via and_masks.

        Args:
            q_len: Query sequence length (number of actual positions)
            kv_len: Key/Value sequence length (should be q_len + 1 for ghost)
            device: Device to create mask on

        Returns:
            Block mask for attention with ghostmax
        """
        cache_key = (q_len, kv_len, str(device))

        if cache_key in self.block_mask_cache:
            return self.block_mask_cache[cache_key]

        block_mask = self.create_block_mask(
            self._build_mask_mod(),
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

        self.block_mask_cache[cache_key] = block_mask
        return block_mask

    def _use_cpu_fallback(self, device: torch.device) -> bool:
        """
        Determine if we should use CPU fallback instead of flex_attention.

        FlexAttention uses torch.compile internally, which doesn't support CPU.

        Args:
            device: The device tensors are on

        Returns:
            True if we should use SDPA fallback, False if we can use flex_attention
        """
        # Use fallback if flex_attention isn't available
        if self.flex_attention is None:
            return True

        # Use fallback on CPU (flex_attention compilation fails on CPU)
        if device.type == "cpu":
            return True

        return False

    def _ghost_aware_attention(
        self,
        q: Tensor,
        k_ghost: Tensor,
        v_ghost: Tensor,
        seq_len: int,
        is_gqa: bool,
    ) -> Tensor:
        """Manual masked attention preserving the ghost column.

        Inputs already include the zero ghost at kv index 0
        (``k_ghost`` / ``v_ghost`` have shape ``[B, H, seq_len+1, D]``).
        Implements softmax1 by leaving the ghost in the softmax denominator
        - the ghost's zero V contributes nothing to the output but adds an
        extra ``exp(0)=1`` term to the sum, exactly matching
        :func:`~praxis.functional.ghostmax`.

        Args:
            q: Query tensor ``[B, num_query_heads, seq_len, head_dim]``.
            k_ghost: Key tensor including ghost at index 0
                ``[B, num_heads, seq_len+1, head_dim]``.
            v_ghost: Value tensor including ghost at index 0.
            seq_len: Number of real (non-ghost) positions.
            is_gqa: Whether to expand K/V across query-head groups.
        """
        device = q.device
        batch_size, _, _, head_dim = q.shape

        if is_gqa:
            k_exp = k_ghost.repeat_interleave(self.num_queries, dim=1)
            v_exp = v_ghost.repeat_interleave(self.num_queries, dim=1)
        else:
            k_exp, v_exp = k_ghost, v_ghost

        scale = 1.0 / (head_dim**0.5)
        scores = (q @ k_exp.transpose(-2, -1)) * scale  # [B, Hq, S, S+1]

        # ALiBi positional bias if not using RoPE.
        if self.pos_type != "rope" and hasattr(self, "alibi_slopes"):
            alibi_bias = self.alibi_slopes.to(device)
            q_pos = torch.arange(seq_len, device=device).unsqueeze(-1)
            kv_pos = torch.arange(seq_len + 1, device=device).unsqueeze(0)
            is_not_ghost = (kv_pos > 0).float()
            actual_kv = kv_pos - 1
            bias = alibi_bias.view(-1, 1, 1) * (actual_kv - q_pos) * is_not_ghost
            scores = scores + bias.unsqueeze(0)

        # Build mask: ghost (kv_idx=0) always reachable; real keys
        # gated by causal + optional sliding window.
        q_pos = torch.arange(seq_len, device=device)
        kv_pos_real = torch.arange(seq_len + 1, device=device) - 1
        if self.causal:
            allowed = q_pos.unsqueeze(-1) >= kv_pos_real.unsqueeze(0)
        else:
            allowed = torch.ones(seq_len, seq_len + 1, dtype=torch.bool, device=device)
        if self.window_size is not None:
            within_window = (
                q_pos.unsqueeze(-1) - kv_pos_real.unsqueeze(0)
            ) <= self.window_size
            allowed = allowed & within_window
        # Ghost column always passes.
        allowed[:, 0] = True
        scores = scores.masked_fill(~allowed.view(1, 1, seq_len, seq_len + 1), -1e9)

        weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0:
            weights = F.dropout(weights, p=self.dropout_p)
        return weights @ v_exp

    def _sdpa_fallback(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = True,
        enable_gqa: bool = False,
    ) -> Tensor:
        """
        Fallback attention using F.scaled_dot_product_attention.

        Used when flex_attention is not available or on CPU.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, kv_len, head_dim]
            v: Value tensor [batch, num_heads, kv_len, head_dim]
            is_causal: Whether to use causal masking

        Returns:
            Attention output [batch, num_heads, seq_len, head_dim]
        """
        # Apply ALiBi bias if needed
        attn_mask = None
        if self.pos_type == "alibi" and hasattr(self, "alibi_slopes"):
            batch_size, num_heads, seq_len, _ = q.shape
            kv_len = k.shape[2]

            # Create position differences matrix
            q_pos = torch.arange(seq_len, device=q.device).unsqueeze(1)
            k_pos = torch.arange(kv_len, device=k.device).unsqueeze(0)
            pos_diff = k_pos - q_pos  # [seq_len, kv_len]

            # Apply ALiBi slopes: [num_heads, 1, 1] * [1, seq_len, kv_len]
            alibi_bias = self.alibi_slopes.to(q.device).view(
                -1, 1, 1
            ) * pos_diff.unsqueeze(0)

            # Expand for batch: [1, num_heads, seq_len, kv_len] -> [batch, num_heads, seq_len, kv_len]
            attn_mask = alibi_bias.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Use PyTorch's scaled_dot_product_attention
        # Note: This handles causal masking, dropout, and is memory-efficient
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
            and attn_mask is None,  # Only use is_causal if no attn_mask
            enable_gqa=enable_gqa,
        )

        return attn_output

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
            attention_mask: Optional mask tensor (currently ignored - use causal masking)
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

        # Note: attention_mask is currently not used with FlexAttention.
        # Causal masking is handled by block_mask.
        # Padding masks could be implemented via score_mod if needed.

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

        # Apply positional encoding (RoPE modifies q,k directly, must happen before ghostmax)
        if self.pos_type == "rope":
            q, k = self._apply_rope(q, k)

        # Ghostmax: Prepend zero token to K and V (after positional encoding)
        # This implements softmax1 by adding an implicit exp(0)=1 to the denominator
        # and a zero vector that contributes nothing to the output.
        # See: https://www.evanmiller.org/attention-is-off-by-one.html
        zero_k = torch.zeros(
            batch_size, self.num_heads, 1, self.head_dim, device=k.device, dtype=k.dtype
        )
        zero_v = torch.zeros(
            batch_size, self.num_heads, 1, self.head_dim, device=v.device, dtype=v.dtype
        )
        k = torch.cat([zero_k, k], dim=2)  # (B, H, T+1, D)
        v = torch.cat([zero_v, v], dim=2)  # (B, H, T+1, D)
        kv_len = seq_len + 1

        # Define score_mod for positional bias (after ghostmax, so it accounts for ghost at kv_idx=0)
        if self.pos_type == "rope":
            score_mod = None
        else:  # alibi
            alibi_bias = self.alibi_slopes.to(inputs.device)

            def alibi_score_mod(score, b, h, q_idx, kv_idx):
                """Apply ALiBi positional bias to attention scores.

                Ghost token at kv_idx=0 receives no bias.
                For actual tokens (kv_idx >= 1), apply standard ALiBi bias
                accounting for the ghost shift: kv_idx=1 corresponds to position 0.

                Uses tensor ops instead of if/else for torch.compile compatibility.
                """
                # For ghost token (kv_idx=0): is_not_ghost=0, so bias=0
                # For actual tokens (kv_idx>=1): is_not_ghost=1, so bias is applied
                is_not_ghost = (kv_idx > 0).float()
                actual_kv = kv_idx - 1
                bias = alibi_bias[h] * (actual_kv - q_idx) * is_not_ghost
                return score + bias

            score_mod = alibi_score_mod

        # Determine if we're using GQA
        is_gqa = self.num_queries > 1

        # Check if we should use CPU fallback
        use_fallback = self._use_cpu_fallback(inputs.device)

        if use_fallback:
            if not hasattr(self, "_cpu_fallback_warned"):
                print(
                    "[CausalAttention] Using manual ghost-aware SDPA fallback "
                    "(CPU device - flex_attention not supported)"
                )
                self._cpu_fallback_warned = True
            # Manual masked attention that preserves the ghost column so
            # softmax1/ghostmax behavior matches the flex_attention path.
            attn_output = self._ghost_aware_attention(
                q, k, v, seq_len=seq_len, is_gqa=is_gqa
            )
        else:
            # Use FlexAttention (GPU only)
            # Handle masking: use causal block_mask with ghost token support
            if self.causal:
                # Create causal block_mask that allows ghost access
                block_mask = self._create_causal_mask(seq_len, kv_len, inputs.device)
            else:
                block_mask = None

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
