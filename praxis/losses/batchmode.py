"""Mode-level losses over the batch's per-token loss distribution.

Batchmean CE is the first moment of a heavily right-skewed distribution: the
mean is dominated by the tail (rare, hard tokens), while the mode - the peak
of the loss DENSITY - is the consensus band. These two criteria treat that
distribution as the object of interest:

``mode_cross_entropy`` (mode-as-target, floored): weight each token by the estimated
density at its own loss value, floored. Consensus-band tokens get full
gradient, tail tokens get FLOOR * lr worth - the "stable static prior plus
tiny baby steps on isolated geometry" hypothesis. The floor is the part that
makes the fringe learnable at all: pure mode-seeking sends zero gradient to
exactly the mass where rare structure lives. Predictions on existing cards:
LLC (RLCT probe) should drop vs a batchmean baseline as the bulk collapses to
a degenerate stable solution; tail CE should plateau, then recover on the
floor's slow timescale. If the tail never recovers, the fringe stayed inert
and the hypothesis fails.

``mode_baseline_cross_entropy`` (mode-as-baseline, the fallback): weight each
token by its deviation ABOVE the modal loss, floored. The consensus band contributes
only floor-level gradient (a DC-blocking filter on the common mode); tokens
the model has not yet made consensus receive full pressure. The
deviation-seeking dual of the above.

Density estimation is endogenous and sync-free: a fixed-bin histogram over the
detached per-token losses, smoothed with a Gaussian kernel of Silverman
bandwidth (the classic 1.06 * std * n^(-1/5) rule; robust variants need a
quantile, which forces a host sync). Everything stays on device; degenerate
batches (uniform losses, all-masked rows) collapse gracefully to the plain
mean via non-finite-weight sanitization, with no data-dependent branching.

Both compose with per-task ``loss_weights`` by multiplication, so e.g. the
preference policy's hard-zeroed PREF_REJECTED positions stay zeroed.
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.losses.reduction import weighted_reduce

# Minimum per-token weight. Fixed, model-agnostic: the same floored-mixture
# rule as the memory bandit, the residual smear, and LionGeo's logit clamp.
# It sets the bulk/fringe timescale separation: fringe tokens learn at
# FLOOR * lr against a stable consensus background.
FLOOR = 0.1
# Histogram resolution for the loss-density estimate. Fixed; the Silverman
# kernel smooths across bins, so the exact count is not sensitive.
BINS = 64


class ModeCrossEntropyLoss(nn.Module):
    """Floored mode-as-target: per-token CE weighted by the loss-density at
    each token's own loss value (consensus band leads, tail keeps FLOOR)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._last: dict = {}

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        input_ids: Optional[Tensor] = None,
        loss_weights: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        per_token = F.cross_entropy(
            flat_logits, flat_labels, reduction="none", ignore_index=-100
        )
        weights = self._mode_weights(per_token.detach(), flat_labels)
        if loss_weights is not None:
            weights = weights * loss_weights.reshape(-1).to(weights.dtype)
        return weighted_reduce(per_token, labels=flat_labels, loss_weights=weights)

    def _mode_weights(self, losses: Tensor, flat_labels: Tensor) -> Tensor:
        """Per-token weights from the smoothed loss-density. Detached inputs;
        no host syncs (degenerate cases sanitize instead of branching)."""
        valid = (flat_labels != -100).to(losses.dtype)
        n = valid.sum().clamp_min(1.0)

        inf = torch.finfo(losses.dtype).max
        lo = torch.where(valid > 0, losses, losses.new_tensor(inf)).amin()
        hi = torch.where(valid > 0, losses, losses.new_tensor(-inf)).amax()
        span = (hi - lo).clamp_min(1e-8)

        idx = ((losses - lo) / span * BINS).long().clamp_(0, BINS - 1)
        counts = torch.bincount(idx, weights=valid, minlength=BINS)

        # Silverman bandwidth over the valid losses, floored at one bin width
        # so the kernel is always well conditioned.
        mean = (losses * valid).sum() / n
        std = (((losses - mean) ** 2 * valid).sum() / n).sqrt()
        h = (1.06 * std * n.pow(-0.2)).clamp_min(span / BINS).clamp_min(1e-8)

        centers = lo + (torch.arange(BINS, device=losses.device) + 0.5) * span / BINS
        kernel = torch.exp(-0.5 * ((centers[:, None] - centers[None, :]) / h) ** 2)
        density_bins = counts @ kernel

        density = density_bins[idx]
        d_max = torch.where(valid > 0, density, density.new_zeros(())).amax()
        mode = centers[density_bins.argmax()]

        weights = self._shape_weights(losses, valid, density, d_max, mode)
        # Degenerate batches (all-masked, zero span) surface as non-finite
        # weights; fall back to the plain mean there instead of branching.
        weights = torch.where(
            torch.isfinite(weights), weights, torch.ones_like(weights)
        )

        # Detached scalars for the dynamics cards; float conversion (the sync)
        # happens only when training_metrics() reads them at the log interval.
        self._last = {
            "mode": mode.detach(),
            "mean_gap": (mean - mode).detach(),
            "weight_mean": ((weights * valid).sum() / n).detach(),
        }
        return weights

    def _shape_weights(self, losses, valid, density, d_max, mode) -> Tensor:
        return FLOOR + (1.0 - FLOOR) * density / d_max.clamp_min(1e-8)

    def training_metrics(self) -> dict:
        if not self._last:
            return {}
        return {
            "batchmode_mode": float(self._last["mode"]),
            "batchmode_mean_gap": float(self._last["mean_gap"]),
            "batchmode_weight_mean": float(self._last["weight_mean"]),
        }

    metric_descriptions = {
        "batchmode_mode": {
            "description": (
                "Modal per-token loss (peak of the smoothed batch loss "
                "density) - the consensus band the mode-level criterion "
                "anchors to."
            ),
            "chart": {
                "title": "Batch Loss Mode",
                "group": "batchmode",
                "group_order": 72,
                "order": 0,
            },
        },
        "batchmode_mean_gap": {
            "description": (
                "Mean loss minus modal loss: the tail mass. Near 0 = the "
                "distribution is unimodal and tight (mode ~ mean, the "
                "criterion is inert); large = a heavy tail the mode-weighting "
                "is actively re-weighting."
            ),
            "chart": {
                "title": "Loss Mean - Mode Gap",
                "group": "batchmode",
                "order": 1,
            },
        },
        "batchmode_weight_mean": {
            "description": (
                "Mean per-token weight over valid tokens, in [FLOOR, 1]. Low "
                "= the density is concentrated and most tokens sit off-peak "
                "(strong selection); near 1 = weighting is nearly uniform."
            ),
            "chart": {"title": "Mode Weight Mean", "group": "batchmode", "order": 2},
        },
    }


class ModeBaselineCrossEntropyLoss(ModeCrossEntropyLoss):
    """Floored mode-as-baseline: weight by deviation ABOVE the modal loss.
    The consensus band is the baseline (floor-level gradient only); tokens
    above it get pressure proportional to their deviation. The fallback dual
    of :class:`ModeCrossEntropyLoss`."""

    def _shape_weights(self, losses, valid, density, d_max, mode) -> Tensor:
        deviation = (losses - mode).clamp_min(0.0) * valid
        return FLOOR + (1.0 - FLOOR) * deviation / deviation.amax().clamp_min(1e-8)
