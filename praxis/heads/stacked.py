"""Stacked head: harmonic field feeding the crystal classifier.

Composes the two mechanisms. ``HarmonicField`` modulates the incoming
features multiplicatively (``h * (1 + b)``), then ``CrystalClassifier``
turns the modulated features into distance-based, min-normalized logits.
One coherent logit stream - harmonics shape the features, crystal
geometry classifies them - so crystal's sign structure (which inference
processors like repetition_penalty depend on) is preserved.

Only the harmonic *field* is borrowed; harmonic's own linear projection
is intentionally not built (the classifier is crystal's). Both auxiliary
losses (smoothness + centers_rms) stay active, and the per-head charts
surface automatically because the field and crystal classifier are
submodules (see ``BaseHead.all_metric_descriptions``).
"""

from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from praxis.heads.base import BaseHead
from praxis.heads.crystal import CrystalHead
from praxis.heads.harmonic import IRR_D, IRR_T, HarmonicField


class StackedHead(BaseHead):
    """Harmonic field stacked in front of the crystal distance classifier."""

    def __init__(self, config: Any, encoder: Optional[nn.Module] = None) -> None:
        super().__init__(config, encoder)

        dims = self.output_dims()
        if dims is None:
            raise ValueError(
                "head_type='crystal_harmonic' needs an encoder that declares "
                "an output layout (same requirement as crystal)."
            )
        feature_dim, _ = dims

        # Field period: same sizing as HarmonicHead. It must cover the
        # positions actually seen; bound max_position_embeddings to block_size
        # in config to keep the field's irfft2 tractable under the byte *8.
        max_positions = int(getattr(config, "max_position_embeddings", 32768) or 32768)
        if "byte" in str(getattr(config, "encoder_type", "")):
            max_positions = max(max_positions, max_positions * 8)

        # Field modulates features at their incoming width; crystal then
        # classifies. Build the field directly (not a full HarmonicHead) so we
        # don't allocate harmonic's unused linear projection.
        self.field = HarmonicField(hidden_dim=feature_dim, max_positions=max_positions)
        self.crystal = CrystalHead(config, encoder=encoder)

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        return self.crystal(self.field(hidden_states))

    @property
    def classifier(self) -> Optional[nn.Module]:
        # The field is applied in forward(); the downstream projection is
        # crystal's distance classifier.
        return self.crystal.classifier

    def aux_losses(self) -> dict:
        out = dict(self.crystal.aux_losses())
        smooth = self.field.aux_loss()
        if smooth is not None:
            out["harmonic_smoothness"] = smooth
        return out

    def training_metrics(self) -> dict:
        out = dict(self.crystal.training_metrics())
        amps = self.field.amplitudes
        out["harmonic_amplitudes_norm"] = float(amps.detach().norm().item())
        out["harmonic_concentration"] = float(self.field.concentration().item())
        out["harmonic_smoothness"] = float(self.field.smoothness().item())
        # Reads whether learning flows into the field or past it into the
        # crystal centers (the downstream classifier here).
        amps_grad = amps.grad
        centers_grad = self.crystal.lm_head.centers.grad
        if amps_grad is not None and centers_grad is not None:
            centers_norm = float(centers_grad.detach().norm().item())
            if centers_norm > 0:
                out["harmonic_grad_ratio"] = (
                    float(amps_grad.detach().norm().item()) / centers_norm
                )
        return out

    def dashboard_snapshots(self) -> dict:
        out = dict(self.crystal.dashboard_snapshots())
        amps = self.field.amplitudes.detach().abs().to("cpu", dtype=torch.float32)
        F_t, F_d = int(amps.shape[0]), int(amps.shape[1])
        out["harmonic_spectrum"] = {
            "grid": amps.tolist(),
            "grid_rows": F_t,
            "grid_cols": F_d,
            "x_range": [1, F_d],
            "y_range": [1, F_t],
            "max_count": float(amps.max().item()) if amps.numel() else 0.0,
            "irrationals": {"t": float(IRR_T), "d": float(IRR_D)},
        }
        return out
