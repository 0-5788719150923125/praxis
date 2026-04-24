"""
InfiniAttention: CausalAttention with segment-level compressive memory.

Processes sequences in segments, computing local attention within each
segment (via CausalAttention mechanics) while maintaining a compressive memory
that accumulates context across segments. Later segments retrieve from
memory to access a summary of all prior segments, providing global context
even with bounded local attention.

Based on "Leave No Context Behind":
https://arxiv.org/abs/2404.07143
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.attention.causal import CausalAttention

_DEFAULT_SEGMENT_SIZE = 256


class InfiniAttention(CausalAttention):
    """
    CausalAttention subclass that adds segment-level compressive memory.

    The sequence is split into segments. Each segment gets local causal
    attention (with ghostmax, RoPE/ALiBi, GQA from the parent). Between
    segments, an ELU+1 kernel memory accumulates key-value context and a
    learned gate blends memory retrieval with local attention output.
    """

    def __init__(self, config) -> None:
        super().__init__(config)

        # Segment size for chunked processing — repurpose window_size,
        # then clear it so the parent uses simple causal masking within
        # each segment (the segmentation itself provides locality).
        self.segment_size = self.window_size or _DEFAULT_SEGMENT_SIZE
        self.window_size = None

        # Override parent's bias-free projections with biased versions
        hidden_size = config.hidden_size
        qkv_dim = (
            self.num_query_heads * self.head_dim + 2 * self.num_heads * self.head_dim
        )
        self.qkv = nn.Linear(hidden_size, qkv_dim, bias=True)
        self.output = nn.Linear(
            self.num_query_heads * self.head_dim, hidden_size, bias=True
        )

        # Learned blending gate
        self.betas = nn.Parameter(
            torch.zeros(1, self.num_query_heads, 1, self.head_dim)
        )

        # Learnable initial memory state
        self.init_mem = nn.Parameter(
            torch.randn(1, self.num_query_heads, self.head_dim, self.head_dim) * 0.01
        )
        self.init_z = nn.Parameter(
            torch.ones(1, self.num_query_heads, self.head_dim, 1) / self.head_dim
        )

    def _init_memory(
        self, batch_size: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Initialize memory from learned parameters."""
        return (
            self.init_mem.expand(batch_size, -1, -1, -1).to(device),
            self.init_z.expand(batch_size, -1, -1, -1).to(device),
        )

    def _retrieve_memory(
        self, q: Tensor, memory_states: Tensor, memory_z: Tensor
    ) -> Tensor:
        """Retrieve from memory using ELU+1 kernel on queries."""
        sigma_q = F.elu(q) + 1.0
        return (sigma_q @ memory_states) / (sigma_q @ memory_z + 1e-6)

    def _update_memory(
        self, k: Tensor, v: Tensor, memory_states: Tensor, memory_z: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Update memory with new keys/values using delta rule."""
        sigma_k = F.elu(k) + 1.0

        # Delta rule: only write what memory doesn't already know
        retrieved = (sigma_k @ memory_states) / (sigma_k @ memory_z + 1e-6)
        value_delta = v - retrieved

        new_states = memory_states + sigma_k.transpose(-2, -1) @ value_delta
        new_z = memory_z + sigma_k.sum(dim=-2, keepdim=True).transpose(-2, -1)
        return new_states, new_z

    def _local_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        seg_len: int,
        device: torch.device,
    ) -> Tensor:
        """Compute local causal attention with ghostmax for one segment."""
        batch_size = q.size(0)

        # Ghostmax: prepend zero token to K and V
        zero_k = torch.zeros(
            batch_size,
            self.num_heads,
            1,
            self.head_dim,
            device=device,
            dtype=k.dtype,
        )
        zero_v = torch.zeros(
            batch_size,
            self.num_heads,
            1,
            self.head_dim,
            device=device,
            dtype=v.dtype,
        )
        k_ghost = torch.cat([zero_k, k], dim=2)
        v_ghost = torch.cat([zero_v, v], dim=2)
        kv_len = seg_len + 1

        # Score modification for ALiBi
        if self.pos_type == "rope":
            score_mod = None
        else:
            alibi_bias = self.alibi_slopes.to(device)

            def alibi_score_mod(score, b, h, q_idx, kv_idx):
                is_not_ghost = (kv_idx > 0).float()
                actual_kv = kv_idx - 1
                bias = alibi_bias[h] * (actual_kv - q_idx) * is_not_ghost
                return score + bias

            score_mod = alibi_score_mod

        is_gqa = self.num_queries > 1
        use_fallback = self._use_cpu_fallback(device)

        if use_fallback:
            if not hasattr(self, "_cpu_fallback_warned"):
                print(
                    "[InfiniAttention] Using SDPA fallback "
                    "(CPU device - flex_attention not supported)"
                )
                self._cpu_fallback_warned = True
            k_local = k_ghost[:, :, 1:, :]
            v_local = v_ghost[:, :, 1:, :]
            return self._sdpa_fallback(
                q, k_local, v_local, is_causal=self.causal, enable_gqa=is_gqa
            )

        if self.causal:
            block_mask = self._create_causal_mask(seg_len, kv_len, device)
        else:
            block_mask = None

        return self.flex_attention(
            q,
            k_ghost,
            v_ghost,
            block_mask=block_mask,
            score_mod=score_mod,
            enable_gqa=is_gqa,
            scale=None,
        )

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        block_ids: Optional[Tensor] = None,
        current_depth: int = 0,
    ) -> Tuple[Tensor, Optional[Tensor], float]:
        batch_size, seq_len, _ = inputs.shape

        # --- Full-sequence QKV projection + positional encoding ---
        qkv = self.qkv(inputs)
        q_dim = self.num_query_heads * self.head_dim
        kv_dim = self.num_heads * self.head_dim

        q = qkv[..., :q_dim]
        k = qkv[..., q_dim : q_dim + kv_dim]
        v = qkv[..., q_dim + kv_dim :]

        q = q.view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.pos_type == "rope":
            q, k = self._apply_rope(q, k)

        # Expand K/V for GQA before memory operations
        if self.num_queries > 1:
            mem_k = k.repeat_interleave(self.num_queries, dim=1)
            mem_v = v.repeat_interleave(self.num_queries, dim=1)
        else:
            mem_k, mem_v = k, v

        # --- Segment-level processing with memory ---
        memory_states, memory_z = self._init_memory(batch_size, q.device)
        gate = torch.sigmoid(self.betas)
        segment_outputs = []

        for start in range(0, seq_len, self.segment_size):
            end = min(start + self.segment_size, seq_len)

            # Slice this segment's Q, K, V
            seg_q = q[:, :, start:end]
            seg_k = k[:, :, start:end]
            seg_v = v[:, :, start:end]
            seg_mem_k = mem_k[:, :, start:end]
            seg_mem_v = mem_v[:, :, start:end]
            seg_len = end - start

            # Retrieve from memory (context from all prior segments)
            memory_output = self._retrieve_memory(seg_q, memory_states, memory_z)

            # Local causal attention within this segment
            attn_output = self._local_attention(
                seg_q, seg_k, seg_v, seg_len, inputs.device
            )

            # Blend memory with local attention
            segment_outputs.append(gate * memory_output + (1 - gate) * attn_output)

            # Update memory with this segment's K/V for subsequent segments
            memory_states, memory_z = self._update_memory(
                seg_mem_k, seg_mem_v, memory_states, memory_z
            )

        # --- Concatenate segments and project ---
        output = torch.cat(segment_outputs, dim=2)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)

        output = self.output(output)
        output = self.dropout(output)

        return output, past_key_values, 0.0
