"""High Error Margin loss, adapted to language modeling."""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.losses.reduction import weighted_reduce

# The paper's reference corpus size: it sets the margin through
# mu = sqrt(M / num_samples). A token stream has no fixed num_samples, so
# Praxis pins the base margin to the point where that formula returns 1.0 -
# which is also what the per-class rule returns under a uniform prior.
REFERENCE_SAMPLES = 2000.0

# Guard against the denominator touching zero in a selective mean; the
# reference implementation uses the same constant.
EPS = 1e-6


class HighErrorMarginLoss(nn.Module):
    """Margin loss on raw logits: every non-target logit is penalized for
    coming within ``margin`` of the target's, and only the above-average
    violations are averaged, so a handful of near-misses still costs as much
    as a lone one. Gradient stops once a token is separated, which is where
    the calibration and continual-learning behavior comes from.

    Set ``adaptive_margin`` for the paper's HEM+ variant: each target's margin
    scales as ``1/sqrt(n * p)`` against a running unigram estimate, widening
    the margin the rarer the token. Costs a dense ``[tokens, vocab]``
    intermediate, like ``mile``.

    From "Margin-based Neural Network Training for Robustness and
    Calibration" (https://arxiv.org/abs/2501.12191).
    """

    metric_descriptions = {
        "hem_separated": {
            "description": (
                "Fraction of tokens already separated by the margin, which therefore "
                "contribute no gradient. Rising toward 1.0 is HEM converging, not "
                "stalling."
            ),
            "chart": {
                "title": "HEM Separated Tokens",
                "y_label": "Fraction",
                "y_scale": "linear",
                "group": "high_error_margin",
                "group_order": 73,
                "order": 0,
            },
        },
        "hem_violators": {
            "description": (
                "Mean number of vocabulary entries per token whose logit sits inside "
                "the margin of the target's."
            ),
            "chart": {
                "title": "HEM Violators per Token",
                "y_label": "Count",
                "y_scale": "log",
                "group": "high_error_margin",
                "order": 10,
            },
        },
        "hem_logit_gap": {
            "description": (
                "Mean target logit minus the largest competing logit. Negative means "
                "the argmax is wrong; HEM only pushes this to ``margin``, so it stays "
                "far below what cross-entropy drives."
            ),
            "chart": {
                "title": "HEM Logit Gap",
                "y_label": "Logits",
                "y_scale": "linear",
                "group": "high_error_margin",
                "order": 20,
            },
        },
        "hem_margin": {
            "description": (
                "Mean margin actually applied. Constant unless the adaptive per-token "
                "variant is running."
            ),
            "chart": {
                "title": "HEM Margin",
                "y_label": "Logits",
                "y_scale": "linear",
                "group": "high_error_margin",
                "order": 30,
            },
        },
    }

    def __init__(
        self,
        margin: float = 1.0,
        adaptive_margin: bool = False,
        margin_ratio_cap: float = 10.0,
        vocab_size: int = 1024,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.adaptive_margin = adaptive_margin
        self.margin_ratio_cap = margin_ratio_cap
        self._metrics: dict = {}
        if adaptive_margin:
            # Laplace prior, so an unseen vocabulary is uniform and every
            # margin starts at the base value rather than at the cap.
            self.register_buffer("token_counts", torch.ones(vocab_size))

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        loss_weights: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Args:
        logits: Predicted logits, already shifted to match labels.
        labels: Target labels; ``-100`` marks a masked position.
        loss_weights: Optional per-token weight tensor matching ``labels``
            shape. See :func:`praxis.losses.reduction.weighted_reduce`.
        """
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        # -100 cannot index a gather; the mask below discards these rows.
        active = flat_labels != -100
        targets = flat_labels.clamp_min(0).unsqueeze(1)

        margin = self._margins(targets, active, flat_logits)
        target_logits = flat_logits.gather(1, targets)
        error = F.relu(flat_logits - (target_logits - margin))
        # The target's own column is not a competitor - the reference zeroes
        # it through a (1 - one_hot) factor.
        error = error.scatter(1, targets, 0.0)

        per_token = self._mean_above_average(error)
        self._record(
            flat_logits, targets, target_logits, margin, error, per_token, active
        )

        # HEM averages only over tokens that still carry error: a separated
        # token is finished, not a zero pulling the mean down.
        weights = (per_token.detach() > 0) & active
        weights = weights.to(per_token.dtype)
        if loss_weights is not None:
            weights = weights * loss_weights.reshape(-1).to(per_token.dtype)
        return weighted_reduce(per_token, loss_weights=weights)

    def _margins(self, targets: Tensor, active: Tensor, logits: Tensor) -> Tensor:
        """Per-token margin, broadcastable over ``[tokens, vocab]``. Constant
        unless HEM+ is on."""
        if not self.adaptive_margin:
            return torch.full(
                (1, 1), self.margin, device=logits.device, dtype=logits.dtype
            )

        counts = self.token_counts
        probabilities = counts / counts.sum()
        # The paper's mu_i = sqrt(M / (n * s_i)), written against the prior so
        # it stays scale-free as the stream grows.
        scale = (probabilities.numel() * probabilities).rsqrt()
        scale = scale.clamp(1.0 / self.margin_ratio_cap, self.margin_ratio_cap)
        margin = self.margin * scale[targets.squeeze(1)].unsqueeze(1)

        with torch.no_grad():
            observed = targets.squeeze(1)[active]
            if observed.numel() > 0:
                counts.index_add_(
                    0, observed, torch.ones_like(observed, dtype=counts.dtype)
                )
        return margin.to(logits.dtype)

    @staticmethod
    def _mean_above_average(error: Tensor) -> Tensor:
        """Mean of each token's errors that sit at or above that token's own
        mean error, counting only the strictly positive ones."""
        threshold = error.detach().mean(dim=1, keepdim=True)
        kept = torch.where(error >= threshold, error, torch.zeros_like(error))
        return kept.sum(dim=1) / (EPS + (kept.detach() > 0).sum(dim=1))

    @torch.no_grad()
    def _record(
        self,
        logits: Tensor,
        targets: Tensor,
        target_logits: Tensor,
        margin: Tensor,
        error: Tensor,
        per_token: Tensor,
        active: Tensor,
    ) -> None:
        """Detached scalars for the dynamics cards. Converted to float only
        when training_metrics() reads them at the log interval."""
        n = active.sum().clamp_min(1)
        runner_up = logits.scatter(1, targets, float("-inf")).amax(dim=1)
        gap = (target_logits.squeeze(1) - runner_up)[active]
        self._metrics = {
            "hem_separated": ((per_token == 0) & active).sum() / n,
            "hem_violators": ((error > 0).sum(dim=1) * active).sum() / n,
            "hem_logit_gap": gap[torch.isfinite(gap)].mean(),
            "hem_margin": margin.expand(active.shape[0], 1).squeeze(1)[active].mean(),
        }

    def training_metrics(self) -> dict:
        """Scalars from the last forward, surfaced to the metrics logger."""
        return {
            key: float(value)
            for key, value in self._metrics.items()
            if torch.isfinite(value)
        }
