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

    # Depth-specialization diagnostics (see praxis.metrics.specialization),
    # averaged across ArcAttention layers and surfaced to the Dynamics tab.
    metric_descriptions = {
        "arc_qkv_specialization": {
            "description": (
                "Depth-specific fraction of the per-depth QKV bias "
                "(1 - ||mean||^2 / mean||row||^2). 0 = every recurrent depth "
                "learned the same bias (collapsed, no benefit over a shared "
                "bias); rising = each depth is specializing its QKV projection."
            ),
            "chart": {
                "title": "Arc Depth Specialization",
                "y_label": "Specialized fraction",
                "y_scale": "linear",
                "group": "arc",
                "group_order": 30,
                "order": 10,
                "series_group": "arc_specialization",
                "series_label": "qkv bias",
            },
        },
        "arc_output_specialization": {
            "description": (
                "Depth-specific fraction of the per-depth output-projection "
                "bias. 0 = collapsed to a shared bias; rising = each recurrent "
                "depth specializes its output bias."
            ),
            "chart": {
                "title": "Arc Depth Specialization",
                "y_label": "Specialized fraction",
                "y_scale": "linear",
                "group": "arc",
                "order": 11,
                "series_group": "arc_specialization",
                "series_label": "output bias",
            },
        },
        "arc_qkv_similarity": {
            "description": (
                "Mean pairwise cosine between the per-depth QKV biases. ~1 = "
                "all depths point the same way (collapsed); falling = bias "
                "directions diverging across depth."
            ),
            "chart": {
                "title": "Arc Depth Similarity",
                "y_label": "Mean pairwise cosine",
                "y_scale": "linear",
                "group": "arc",
                "order": 20,
                "series_group": "arc_similarity",
                "series_label": "qkv bias",
            },
        },
        "arc_output_similarity": {
            "description": (
                "Mean pairwise cosine between the per-depth output biases. "
                "~1 = collapsed to one direction; falling = diverging."
            ),
            "chart": {
                "title": "Arc Depth Similarity",
                "y_label": "Mean pairwise cosine",
                "y_scale": "linear",
                "group": "arc",
                "order": 21,
                "series_group": "arc_similarity",
                "series_label": "output bias",
            },
        },
    }

    def __init__(self, config, attention_gating: bool = True, **kwargs) -> None:
        super().__init__(config, **kwargs)

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

    # The forward pass lives in InfiniAttention (including the cached decode
    # path); Arc only customizes these hooks.

    def _project_qkv(self, inputs: Tensor, current_depth: int) -> Tensor:
        depth_idx = torch.tensor(current_depth, device=inputs.device)
        return self.qkv(inputs) + self.depth_qkv_bias(depth_idx)

    def _adjust_kv(
        self, k: Tensor, v: Tensor, current_depth: int
    ) -> Tuple[Tensor, Tensor]:
        # Dropoff ablation: optionally withhold the causal tip at one depth
        # step (inherited from CausalAttention; no-op unless dropoff_step set).
        return self._maybe_dropoff(k, v, current_depth)

    def _finalize_output(
        self, output: Tensor, inputs: Tensor, current_depth: int
    ) -> Tensor:
        if self.attention_gating:
            output = output * torch.sigmoid(self.gate(inputs))
        depth_idx = torch.tensor(current_depth, device=inputs.device)
        return self.output(output) + self.depth_output_bias(depth_idx)

    def training_metrics(self) -> dict:
        """Whether the per-depth biases are specializing or collapsing."""
        from praxis.metrics.specialization import depth_dispersion

        out = {}
        qkv = depth_dispersion(self.depth_qkv_bias.weight)
        if qkv is not None:
            out["arc_qkv_specialization"] = qkv["specialization"]
            out["arc_qkv_similarity"] = qkv["similarity"]
        outp = depth_dispersion(self.depth_output_bias.weight)
        if outp is not None:
            out["arc_output_specialization"] = outp["specialization"]
            out["arc_output_similarity"] = outp["similarity"]
        return out
