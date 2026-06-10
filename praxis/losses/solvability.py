import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Fixed, model-agnostic constants - not knobs to tune per run.
# The credence is read off the first quarter of the trunk sequence: enough
# context to judge the instance, early enough to count as a prediction.
PREFIX_FRACTION = 0.25
# Decay of the running per-sample loss baseline that defines "solved".
EMA_DECAY = 0.99


class SolvabilityProbe(nn.Module):
    """Self-predicted solvability: the model's own credence that it will
    "solve" each sample - beat its running per-sample loss baseline - scored
    against the realized outcome.

    The bounded-observer counterpart of a P/NP grade (see the paper's
    observer-relative tractability section): tractability here is a continuous,
    per-instance credence rather than a class membership. Purely observational:
    the pooled trunk states are detached, so gradients reach only the probe and
    never steer the model being observed.
    """

    # Chart hints for the values training_metrics() produces, kept beside
    # them so both edit in one place. Surfaced to the Dynamics tab manifest.
    metric_descriptions = {
        "solvability_confidence": {
            "description": (
                "Mean self-predicted credence (from an early-prefix probe) "
                "that the model will beat its own running loss baseline on "
                "each sample. The observer-relative solvability grade."
            ),
            "chart": {
                "title": "Self-Predicted Solvability",
                "y_label": "Rate",
                "y_scale": "linear",
                "group": "solvability",
                "order": 10,
                "series_group": "solvability_cal",
                "series_label": "confidence",
            },
        },
        "solvability_solve_rate": {
            "description": (
                "Fraction of samples whose realized loss beat the running "
                "baseline. The outcome the confidence is scored against; "
                "hovers near 0.5 by construction as the baseline tracks."
            ),
            "chart": {
                "title": "Self-Predicted Solvability",
                "y_label": "Rate",
                "y_scale": "linear",
                "group": "solvability",
                "order": 20,
                "series_group": "solvability_cal",
                "series_label": "solve rate",
            },
        },
        "solvability_brier": {
            "description": (
                "Brier score of the solvability credence against the realized "
                "outcome. 0.25 = uninformative; falling means the model "
                "increasingly knows, from an early read, what it can do."
            ),
            "chart": {
                "title": "Solvability Calibration (Brier)",
                "y_label": "Brier Score",
                "y_scale": "linear",
                "group": "solvability",
                "order": 30,
            },
        },
    }

    def __init__(self, hidden_size: int):
        super().__init__()
        inner = max(8, hidden_size // 4)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Linear(inner, 1),
        )
        # Running per-sample loss baseline; checkpointed with the model.
        self.register_buffer("loss_ema", torch.zeros(()))
        self.register_buffer("ema_steps", torch.zeros((), dtype=torch.long))
        self._metrics: dict = {}

    def forward(self, hidden_states: Tensor, logits: Tensor, labels: Tensor) -> Tensor:
        """BCE on the probe's credence vs. the realized per-sample outcome.

        Args:
            hidden_states: Trunk states [B, T, D] (pre-head); detached here.
            logits: Token logits, full-length or already label-aligned.
            labels: Next-token labels (may contain -100).
        """
        # Realized outcome, no grad: mean per-token CE per sample vs. the EMA.
        with torch.no_grad():
            if logits.size(1) == labels.size(1) + 1:
                logits = logits[..., :-1, :]
            per_token = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).view(labels.size(0), -1)
            mask = (labels != -100).float()
            sample_loss = (per_token * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
            batch_mean = sample_loss.mean()
            baseline = self.loss_ema if self.ema_steps > 0 else batch_mean
            solved = (sample_loss < baseline).float()
            self.loss_ema.copy_(
                batch_mean
                if self.ema_steps == 0
                else EMA_DECAY * self.loss_ema + (1 - EMA_DECAY) * batch_mean
            )
            self.ema_steps += 1

        # The prediction: pooled early prefix, detached - observe, don't steer.
        k = max(1, int(hidden_states.size(1) * PREFIX_FRACTION))
        pooled = hidden_states[:, :k].detach().float().mean(dim=1)
        credence_logit = self.proj(pooled).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(credence_logit, solved)

        # Own our diagnostics here, beside the math that produces them.
        with torch.no_grad():
            credence = torch.sigmoid(credence_logit)
            self._metrics = {
                "solvability_confidence": float(credence.mean()),
                "solvability_solve_rate": float(solved.mean()),
                "solvability_brier": float(((credence - solved) ** 2).mean()),
            }
        return loss

    def training_metrics(self) -> dict:
        """Scalars from the last forward, surfaced to the metrics logger."""
        return dict(self._metrics)
