"""
ArcAttention: InfiniAttention with per-depth learned biases.

Adds nn.Embedding(depth, dim) biases to all four projections (Q, K, V, O),
looked up via current_depth. Requires SandwichNorm (post-normalization) to
bound outputs between recurrent depth passes, preventing the compounding
that destabilized earlier per-depth bias attempts.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from praxis.attention.infini import InfiniAttention


class ArcAttention(InfiniAttention):
    """
    InfiniAttention subclass that adds per-depth learned biases.

    Each recurrent depth pass gets its own additive bias on Q, K, V, and O
    projections via nn.Embedding lookups. Zero-initialized so the model
    starts identical to InfiniAttention and gradually specializes.

    Optionally applies head-specific elementwise sigmoid gating to the SDPA
    output (Qiu et al. 2025, arXiv:2505.06708), which introduces non-linearity
    and input-dependent sparsity between the attention output and W_o.
    """

    def __init__(self, config, attention_gating: bool = True) -> None:
        super().__init__(config)

        self.depth = config.depth

        # Per-depth biases for QKV projection (applied after the linear)
        qkv_dim = (
            self.num_query_heads * self.head_dim + 2 * self.num_heads * self.head_dim
        )
        self.depth_qkv_bias = nn.Embedding(self.depth, qkv_dim)
        nn.init.zeros_(self.depth_qkv_bias.weight)

        # Per-depth bias for output projection
        out_dim = config.hidden_size
        self.depth_output_bias = nn.Embedding(self.depth, out_dim)
        nn.init.zeros_(self.depth_output_bias.weight)

        # Head-specific elementwise sigmoid gate on SDPA output.
        # Score shape: (n, num_query_heads * head_dim), computed from input X.
        self.attention_gating = attention_gating
        if self.attention_gating:
            gate_dim = self.num_query_heads * self.head_dim
            self.gate = nn.Linear(config.hidden_size, gate_dim, bias=True)

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        block_ids: Optional[Tensor] = None,
        current_depth: int = 0,
    ) -> Tuple[Tensor, Optional[Tensor], float]:
        batch_size, seq_len, _ = inputs.shape

        # --- Full-sequence QKV projection + per-depth bias ---
        qkv = self.qkv(inputs)
        depth_idx = torch.tensor(current_depth, device=inputs.device)
        qkv = qkv + self.depth_qkv_bias(depth_idx)

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

            seg_q = q[:, :, start:end]
            seg_k = k[:, :, start:end]
            seg_v = v[:, :, start:end]
            seg_mem_k = mem_k[:, :, start:end]
            seg_mem_v = mem_v[:, :, start:end]
            seg_len = end - start

            memory_output = self._retrieve_memory(seg_q, memory_states, memory_z)

            attn_output = self._local_attention(
                seg_q, seg_k, seg_v, seg_len, inputs.device
            )

            segment_outputs.append(gate * memory_output + (1 - gate) * attn_output)

            memory_states, memory_z = self._update_memory(
                seg_mem_k, seg_mem_v, memory_states, memory_z
            )

        # --- Concatenate segments, project, and apply per-depth output bias ---
        output = torch.cat(segment_outputs, dim=2)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)

        if self.attention_gating:
            output = output * torch.sigmoid(self.gate(inputs))

        output = self.output(output)
        output = output + self.depth_output_bias(depth_idx)
        output = self.dropout(output)

        return output, past_key_values, 0.0
