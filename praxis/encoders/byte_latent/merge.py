"""How the local decoder combines the byte path with the trunk's patch output.

``add`` is BLT's sum, ``h = h_encoder + patch_embeds``: nothing holds the two on
one scale, so whichever path grows faster decides the share.

``normalized`` RMS-normalizes both streams, with no learnable scale so neither
can outgrow the other, and adds them at equal weight. Nothing in it can shut a
path off, so both keep their gradient; the decoder's own layers learn what to
read from each.

``gated`` blends the normalized streams with a per-position softmax gate over
the pair. The gate is convex, so the two streams compete and it can saturate
onto one of them, after which the other receives no gradient.

Each position reads only its own two vectors, so every mode is causal.

Both modes report the same magnitudes, so runs on either can be compared.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

MERGES = ("add", "normalized", "gated")


class PatchMerge(nn.Module):
    """``add`` (the BLT reference), or a sum or gated blend of normalized streams."""

    metric_descriptions = {
        "merge_trunk_ratio": {
            "description": (
                "RMS of the trunk's contribution over RMS of the byte path's, where "
                "the local decoder merges them. 1 = equal footing; ~0.01 = the trunk "
                "is a rounding error on the byte path."
            ),
            "chart": {
                "title": "Trunk vs Byte Path at the Merge",
                "y_label": "trunk RMS / byte RMS",
                "y_scale": "logarithmic",
                "group": "byte_merge",
                "group_order": 73,
                "order": 10,
                "series_group": "merge_ratio_pair",
                "series_label": "whole contribution",
            },
        },
        "merge_trunk_content_ratio": {
            "description": (
                "The same ratio for the trunk's INPUT-DEPENDENT part - its "
                "contribution minus its mean over the batch - which is all a "
                "constant offset cannot carry."
            ),
            "chart": {
                "title": "Trunk vs Byte Path at the Merge",
                "y_label": "trunk RMS / byte RMS",
                "y_scale": "logarithmic",
                "group": "byte_merge",
                "order": 11,
                "series_group": "merge_ratio_pair",
                "series_label": "input-dependent part",
            },
        },
        "merge_byte_rms": {
            "description": "RMS of the byte path's contribution at the merge.",
            "chart": {
                "title": "Merge Magnitudes",
                "y_label": "RMS",
                "y_scale": "logarithmic",
                "group": "byte_merge",
                "order": 20,
                "series_group": "merge_rms_pair",
                "series_label": "byte path",
            },
        },
        "merge_trunk_rms": {
            "description": "RMS of the trunk's contribution at the merge.",
            "chart": {
                "title": "Merge Magnitudes",
                "y_label": "RMS",
                "y_scale": "logarithmic",
                "group": "byte_merge",
                "order": 21,
                "series_group": "merge_rms_pair",
                "series_label": "trunk",
            },
        },
        "merge_gate_trunk": {
            "description": (
                "Mean gate weight on the trunk (gated merge only). 0.5 at init; "
                "falling toward 0 = the decoder is choosing to ignore the trunk."
            ),
            "chart": {
                "title": "Merge Gate: Trunk Weight",
                "y_label": "weight",
                "y_scale": "linear",
                "group": "byte_merge",
                "order": 30,
            },
        },
        "merge_gate_entropy": {
            "description": (
                "Mean entropy of the per-position gate (gated merge only). ln 2 "
                "(0.69) = an even split everywhere; 0 = every position picks one "
                "path outright."
            ),
            "chart": {
                "title": "Merge Gate Entropy",
                "y_label": "nats",
                "y_scale": "linear",
                "group": "byte_merge",
                "order": 31,
            },
        },
    }

    def __init__(self, dim: int, mode: str = "add", eps: float = 1e-6) -> None:
        super().__init__()
        if mode not in MERGES:
            raise ValueError(f"Unknown merge {mode!r}; known: {list(MERGES)}")
        self.mode = mode
        self.eps = eps
        if mode == "gated":
            self.gate = nn.Linear(2 * dim, 2, bias=False)
            nn.init.zeros_(self.gate.weight)
        self._stats: Optional[Dict[str, Tensor]] = None

    def extra_repr(self) -> str:
        return f"mode={self.mode}"

    def forward(self, byte: Tensor, trunk: Tensor) -> Tensor:
        if self.mode == "add":
            if self.training:
                self._record(byte, trunk, None)
            return byte + trunk
        dim = (byte.shape[-1],)
        b = F.rms_norm(byte, dim, eps=self.eps)
        t = F.rms_norm(trunk, dim, eps=self.eps)
        if self.mode == "normalized":
            if self.training:
                self._record(b, t, None)
            return b + t
        weights = torch.softmax(self.gate(torch.cat([b, t], dim=-1)), dim=-1)
        byte_part = weights[..., :1] * b
        trunk_part = weights[..., 1:] * t
        if self.training:
            self._record(byte_part, trunk_part, weights)
        return byte_part + trunk_part

    @torch.compiler.disable
    @torch.no_grad()
    def _record(self, byte: Tensor, trunk: Tensor, weights: Optional[Tensor]) -> None:
        """Keep the last training forward's magnitudes as device tensors; the
        host reads them only when the dashboard polls ``training_metrics``."""
        byte, trunk = byte.float(), trunk.float()
        lead = tuple(range(trunk.dim() - 1))
        stats = {
            "byte": byte.pow(2).mean().sqrt(),
            "trunk": trunk.pow(2).mean().sqrt(),
            "content": (trunk - trunk.mean(dim=lead, keepdim=True))
            .pow(2)
            .mean()
            .sqrt(),
        }
        if weights is not None:
            w = weights.float().reshape(-1, 2)
            p = w.clamp_min(1e-9)
            stats["gate_trunk"] = w[:, 1].mean()
            stats["gate_entropy"] = (-(p * p.log()).sum(dim=-1)).mean()
        self._stats = stats

    def training_metrics(self) -> dict:
        stats = self._stats
        if not stats:
            return {}
        byte = max(float(stats["byte"]), 1e-12)
        out = {
            "merge_trunk_ratio": float(stats["trunk"]) / byte,
            "merge_trunk_content_ratio": float(stats["content"]) / byte,
            "merge_byte_rms": float(stats["byte"]),
            "merge_trunk_rms": float(stats["trunk"]),
        }
        if "gate_trunk" in stats:
            out["merge_gate_trunk"] = float(stats["gate_trunk"])
            out["merge_gate_entropy"] = float(stats["gate_entropy"])
        return out
