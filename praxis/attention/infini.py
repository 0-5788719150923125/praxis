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

# One-shot NaN/inf detection inside the compressive-memory path. Helps
# pinpoint whether memory state explodes between segments. Remove once
# the Infini/Arc training stability story is settled.
_MEMORY_DIAG_FIRED = False


def _check_memory_finite(name: str, **tensors: Tensor) -> None:
    """Print a one-shot diagnostic if any tensor contains NaN/inf.

    Intentionally cheap (single ``.isfinite().all()`` per tensor) and
    bounded (only the first hit prints, then it's a no-op). The values
    inspected are the actual memory-update inputs/outputs so the print
    points directly at where the explosion started.
    """
    global _MEMORY_DIAG_FIRED
    if _MEMORY_DIAG_FIRED:
        return
    for label, t in tensors.items():
        if not torch.isfinite(t).all():
            _MEMORY_DIAG_FIRED = True
            n_nan = torch.isnan(t).sum().item()
            n_inf = torch.isinf(t).sum().item()
            finite = t[torch.isfinite(t)]
            max_abs = finite.abs().max().item() if finite.numel() > 0 else float("nan")
            print(
                f"[InfiniMem #diag] non-finite detected in {name}: "
                f"tensor={label} shape={tuple(t.shape)} "
                f"nan={n_nan} inf={n_inf} max_abs_finite={max_abs:.4e}"
            )
            # Also report companions so we can see which input caused it.
            for other_label, other_t in tensors.items():
                if other_label == label:
                    continue
                ot_finite = other_t[torch.isfinite(other_t)]
                ot_max = (
                    ot_finite.abs().max().item()
                    if ot_finite.numel() > 0
                    else float("nan")
                )
                print(
                    f"  - companion {other_label} shape={tuple(other_t.shape)} "
                    f"max_abs_finite={ot_max:.4e}"
                )
            return


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

        # Initial memory state. Registered as buffers (no grad) to match
        # the reference Infini-Attention - memory init shouldn't accumulate
        # gradients through the recurrent across-segment chain, which would
        # turn the segment loop into an exploding-gradient RNN.
        self.register_buffer(
            "init_mem",
            torch.zeros(1, self.num_query_heads, self.head_dim, self.head_dim),
        )
        self.register_buffer(
            "init_z",
            torch.zeros(1, self.num_query_heads, self.head_dim, 1),
        )

    def _init_memory(
        self, batch_size: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Initialize memory from zero-valued buffers (no grad)."""
        return (
            self.init_mem.expand(batch_size, -1, -1, -1).to(device),
            self.init_z.expand(batch_size, -1, -1, -1).to(device),
        )

    def _retrieve_memory(
        self, q: Tensor, memory_states: Tensor, memory_z: Tensor
    ) -> Tensor:
        """Retrieve from memory using ELU+1 kernel on queries."""
        sigma_q = F.elu(q) + 1.0
        out = (sigma_q @ memory_states) / (sigma_q @ memory_z + 1e-6)
        _check_memory_finite(
            "_retrieve_memory",
            sigma_q=sigma_q,
            memory_states=memory_states,
            memory_z=memory_z,
            out=out,
        )
        return out

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
        _check_memory_finite(
            "_update_memory",
            sigma_k=sigma_k,
            memory_states=memory_states,
            memory_z=memory_z,
            value_delta=value_delta,
            new_states=new_states,
            new_z=new_z,
        )
        return new_states, new_z

    def _local_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        seg_len: int,
        device: torch.device,
        seg_block_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute local causal attention with ghostmax for one segment.

        When ``seg_block_ids`` is provided, attention is additionally
        gated so queries can only attend to keys in the same block - this
        keeps packed documents from leaking across EOS boundaries inside
        a segment. The fast flex_attention path doesn't carry block_ids
        through its cached mask, so the block-aware branch drops to a
        manual masked-SDPA implementation.
        """
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

        if seg_block_ids is not None:
            return self._local_attention_blocked(
                q, k_ghost, v_ghost, seg_len, device, seg_block_ids
            )

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
                    "[InfiniAttention] Using manual masked-SDPA fallback "
                    "(CPU device - flex_attention not supported)"
                )
                self._cpu_fallback_warned = True
            # Reuse the block-aware path with a dummy single-block mask so
            # ghostmax (softmax1 via the zero ghost column) is preserved.
            # The previous code stripped the ghost before SDPA, which
            # silently downgraded to plain softmax in the CPU path.
            dummy_blocks = torch.zeros(
                batch_size, seg_len, device=device, dtype=torch.long
            )
            return self._local_attention_blocked(
                q, k_ghost, v_ghost, seg_len, device, dummy_blocks
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

    def _local_attention_blocked(
        self,
        q: Tensor,
        k_ghost: Tensor,
        v_ghost: Tensor,
        seg_len: int,
        device: torch.device,
        seg_block_ids: Tensor,
    ) -> Tensor:
        """Manual local attention that honors per-segment block_ids.

        Ghost token at ``kv_idx == 0`` is always reachable (preserving the
        ghostmax / softmax1 behavior); real keys are reachable only when
        same-block AND causal.
        """
        batch_size, _, _, head_dim = q.shape

        # Expand K, V for GQA so we can multiply directly.
        if self.num_queries > 1:
            k_exp = k_ghost.repeat_interleave(self.num_queries, dim=1)
            v_exp = v_ghost.repeat_interleave(self.num_queries, dim=1)
        else:
            k_exp, v_exp = k_ghost, v_ghost

        scale = 1.0 / (head_dim**0.5)
        scores = (q @ k_exp.transpose(-2, -1)) * scale  # [B, Hq, S, S+1]

        # ALiBi bias (only when not using RoPE).
        if self.pos_type != "rope":
            alibi_bias = self.alibi_slopes.to(device)  # [Hq]
            q_pos = torch.arange(seg_len, device=device).unsqueeze(-1)  # [S, 1]
            kv_pos = torch.arange(seg_len + 1, device=device).unsqueeze(0)  # [1, S+1]
            is_not_ghost = (kv_pos > 0).float()
            actual_kv = kv_pos - 1
            bias = alibi_bias.view(-1, 1, 1) * (actual_kv - q_pos) * is_not_ghost
            scores = scores + bias.unsqueeze(0)  # [1, Hq, S, S+1]

        # Build mask: ghost (kv_idx=0) always allowed; real keys must be
        # in the same block as the query AND causal.
        q_pos = torch.arange(seg_len, device=device)
        kv_pos_real = torch.arange(seg_len + 1, device=device) - 1  # -1 = ghost
        causal = q_pos.unsqueeze(-1) >= kv_pos_real.unsqueeze(0)  # [S, S+1]
        # Pad block_ids on the left with a sentinel for the ghost slot.
        sentinel = torch.full(
            (batch_size, 1), -1, device=device, dtype=seg_block_ids.dtype
        )
        kv_block_ids = torch.cat([sentinel, seg_block_ids], dim=1)  # [B, S+1]
        same_block = seg_block_ids.unsqueeze(-1) == kv_block_ids.unsqueeze(-2)
        # Ghost is always reachable regardless of block.
        ghost_col = torch.zeros(seg_len + 1, dtype=torch.bool, device=device)
        ghost_col[0] = True
        allowed = ghost_col.view(1, 1, -1) | (
            same_block & causal.unsqueeze(0)
        )  # [B, S, S+1]
        allowed = allowed.unsqueeze(1)  # [B, 1, S, S+1] - broadcast over heads
        scores = scores.masked_fill(~allowed, -1e9)

        weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0:
            weights = F.dropout(weights, p=self.dropout_p)
        return weights @ v_exp  # [B, Hq, S, head_dim]

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
            seg_block_ids = block_ids[:, start:end] if block_ids is not None else None

            # Retrieve from memory (context from all prior segments)
            memory_output = self._retrieve_memory(seg_q, memory_states, memory_z)

            # Local causal attention within this segment
            attn_output = self._local_attention(
                seg_q,
                seg_k,
                seg_v,
                seg_len,
                inputs.device,
                seg_block_ids=seg_block_ids,
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
