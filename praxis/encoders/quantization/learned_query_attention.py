"""
Learned query cross-attention for segment-local pooling.

Ported from abstractinator with adaptations for praxis:
- try/except for flex_attention import, matching BLT_ALLOW_MISSING_FLEX_ATTENTION pattern
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Match BLT's pattern for optional flex_attention
_HAS_FLEX_ATTENTION = False
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    _HAS_FLEX_ATTENTION = True
except ImportError:
    if not os.environ.get("BLT_ALLOW_MISSING_FLEX_ATTENTION"):
        raise


class LearnedQueryAttention(nn.Module):
    """
    Cross-attention where the queries are learned inside the module.

    - Owns a learnable template of L queries (shape: L x D).
    - At runtime, repeats that template across a fixed Q_max "query slots".
      slot q maps to segment = q // L and template row = q % L.
    - Segment-local pooling is enforced via a block mask (flex) or by masking scores.

    Args:
        embed_dim: model dimension D
        num_queries_per_segment: L (queries per segment)
        max_queries: Q_max (static total query slots returned)
        num_heads: attention heads
        use_flex_attention: use torch.nn.attention.flex_attention when True
    """

    def __init__(
        self,
        embed_dim: int,
        num_queries_per_segment: int,
        max_queries: int,
        num_heads: int,
        use_flex_attention: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.d_model = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads

        # queries per segment (L) and static output length (Q_max)
        self.L = int(num_queries_per_segment)
        self.Q_max = int(max_queries)
        assert self.Q_max > 0 and self.L > 0
        assert (self.Q_max % self.L) == 0, "Q_max must be a multiple of L"

        # learned query template: (L, D)
        self.query_template = nn.Parameter(torch.randn(self.L, self.d_model))

        # projections & norms
        self.in_norm = nn.RMSNorm(self.d_model)
        self.out_norm = nn.RMSNorm(self.d_model)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.normal_(self.query_template, std=0.02)

        self.use_flex_attention = bool(use_flex_attention) and _HAS_FLEX_ATTENTION

    def _owned_queries(
        self, B: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build (B, Q_max, D) query tensor by tiling the L-row template.
        slot q uses template row (q % L).
        """
        slot = torch.arange(self.Q_max, device=device) % self.L  # (Q_max,)
        q_once = self.query_template[slot].to(dtype=dtype)  # (Q_max, D)
        return q_once.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, Q_max, D)

    @staticmethod
    def _num_segments(
        seg_id: torch.Tensor, key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        nseg[b] = #segments present in real (non-pad) keys of row b.
        seg_id must be contiguous non-decreasing integers starting at 0.
        """
        if key_padding_mask is not None:
            kseg_real = torch.where(
                key_padding_mask, seg_id.new_full((), -1), seg_id
            )
            return (kseg_real.amax(dim=1).clamp(min=-1) + 1).to(
                torch.long
            )  # (B,)
        return (seg_id.amax(dim=1) + 1).to(torch.long)

    def forward(
        self,
        x: torch.Tensor,  # (B, S, D)  keys/values
        seg_id: torch.Tensor,  # (B, S)     segment id per key position
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, S) True==pad
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pooled: (B, Q_max, D)
            attn_weights: (empty) or (B, Q_max, S) if return_attn=False/True
        """
        B, S, D = x.shape
        assert D == self.d_model, f"x dim {D} != model dim {self.d_model}"

        # Build static-shape queries that we own
        queries = self._owned_queries(B, x.device, x.dtype)  # (B, Q_max, D)
        x_norm = self.in_norm(x)

        # projections and reshape for MH attention
        q_proj = self.q_proj(queries)  # (B, Q_max, D)
        k_proj = self.k_proj(x_norm)  # (B, S, D)
        v_proj = self.v_proj(x_norm)  # (B, S, D)

        q = q_proj.view(B, self.Q_max, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (B,H,Q,d)
        k = k_proj.view(B, S, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (B,H,S,d)
        v = v_proj.view(B, S, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (B,H,S,d)

        # How many segments are real in each batch row?
        nseg = self._num_segments(seg_id, key_padding_mask)  # (B,)

        if self.use_flex_attention:
            from torch.nn.attention.flex_attention import (
                create_block_mask,
                flex_attention,
            )

            pad = (
                key_padding_mask.to(torch.bool).contiguous()
                if key_padding_mask is not None
                else None
            )
            kseg = seg_id.to(torch.long).contiguous()

            L = self.L
            Q_max = self.Q_max

            Bn, Hn, Qn, Sn = q.size(0), q.size(1), q.size(2), k.size(2)
            assert Qn == Q_max and Q_max % L == 0

            def keep(b, h, qidx, kidx):
                valid_q = qidx < nseg[b] * L
                qseg = torch.div(qidx, L, rounding_mode="floor")
                same_seg = kseg[b, kidx].eq(qseg)
                if pad is not None:
                    not_pad = ~pad[b, kidx]
                else:
                    not_pad = qidx.eq(qidx)  # always True, stays a tensor
                return valid_q & same_seg & not_pad

            block_mask = create_block_mask(
                keep,
                B=Bn,
                H=Hn,
                Q_LEN=Qn,
                KV_LEN=Sn,
                BLOCK_SIZE=128,
                device=q.device,
            )
            ctx = flex_attention(q, k, v, block_mask=block_mask)
            attn_weights = queries.new_empty(0) if not return_attn else None

        else:
            # Fallback: standard attention with segment gating on scores
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,Q,S)

            if key_padding_mask is not None:
                scores = scores.masked_fill(
                    key_padding_mask[:, None, None, :], float("-inf")
                )

            # segment mismatch mask: (B,Q,S)
            seg_q = (torch.arange(self.Q_max, device=x.device) // self.L)[
                None, :
            ]  # (1,Q)
            mismatch = seg_id[:, None, :] != seg_q[:, :, None]  # (B,Q,S)
            scores = scores.masked_fill(mismatch[:, None, :, :], float("-inf"))

            # drop invalid trailing queries: q >= nseg[b]*L
            valid_q = torch.arange(self.Q_max, device=x.device)[None, :] < (
                nseg * self.L
            )[:, None]  # (B,Q)
            scores = scores.masked_fill(~valid_q[:, None, :, None], float("-inf"))

            attn = torch.softmax(scores, dim=-1)  # (B,H,Q,S)
            ctx = torch.matmul(attn, v)  # (B,H,Q,d)
            attn_weights = (
                attn.mean(dim=1) if return_attn else queries.new_empty(0)
            )

        out = ctx.permute(0, 2, 1, 3).reshape(B, self.Q_max, self.d_model)
        out = self.out_norm(out)
        out = self.out_proj(out)

        # Zero out invalid query slots so downstream components can be statically-shaped
        valid_q_mask = torch.arange(self.Q_max, device=x.device)[None, :] < (
            nseg * self.L
        )[:, None]  # (B,Q)
        out = out * valid_q_mask.to(out.dtype).unsqueeze(-1)

        return out, (attn_weights if return_attn else out.new_empty(()))
