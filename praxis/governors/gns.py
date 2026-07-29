"""Gradient-noise-scale batch governor: the measurement core.

Batch size is the rare hyperparameter with a directly measurable optimal
value. McCandlish et al. 2018 ("An Empirical Model of Large-Batch Training")
show the *gradient noise scale* ``B_noise = S / |G|^2`` - the trace of the
per-sample gradient covariance over the squared true-gradient norm - marks
the batch size where training transitions from noise-dominated (bigger
batches are near-free progress) to signal-dominated (bigger batches waste
data). B_noise grows as the loss falls, so a fixed target_batch_size is
wrong at both ends of a run; tracking it yields the small-early/large-late
schedule ("Don't Decay the Learning Rate, Increase the Batch Size") from an
endogenous signal instead of a tuned curve.

Gradient accumulation hands us the paper's two-point estimator (Appendix A)
for free: the gradient after the FIRST microbatch backward is an estimate at
batch ``B_small = batch_size``; the fully accumulated gradient is an estimate
at ``B_big = K * batch_size``. Two squared norms per optimizer step - no
extra forwards, no extra memory:

    |G|^2_est = (B_big*|G_big|^2 - B_small*|G_small|^2) / (B_big - B_small)
    S_est     = (|G_small|^2 - |G_big|^2) / (1/B_small - 1/B_big)

Both are unbiased but noisy (S_est can even go negative on a lucky draw), so
we keep separate EMAs of numerator and denominator and take the ratio of
EMAs - the estimator the paper recommends - and only report a noise scale
once both EMAs are warm and positive.

All constants are fixed and model-agnostic (no per-experiment tuning; see
``feedback_no_hyperparameter_tuning``). The Lightning wiring lives in
``praxis/callbacks/lightning/governor.py``.
"""

import math
from typing import Any, Dict, Optional

import torch


class GradientNoiseEstimator:
    """Two-point gradient noise scale estimator (McCandlish et al., App. A).

    ``update`` accepts 0-dim tensors and performs only tensor arithmetic, so
    the per-step cost adds no host/device sync; ``noise_scale`` is the one
    call that syncs, intended for the (rarer) decision cadence.
    """

    # Fixed, model-agnostic constants.
    ema_alpha = 0.05  # ~20-observation memory; slow enough to tame S_est noise
    min_updates = 8  # observations before the ratio is trusted at all

    def __init__(self) -> None:
        self._g_sq_ema: Optional[torch.Tensor] = None  # EMA of |G|^2 estimates
        self._s_ema: Optional[torch.Tensor] = None  # EMA of S estimates
        self._updates: int = 0

    def update(self, small_sq, big_sq, b_small: float, b_big: float) -> None:
        """Fold in one optimizer step's pair of squared gradient norms.

        ``small_sq``/``big_sq`` are ``|G_small|^2`` and ``|G_big|^2`` at
        batch sizes ``b_small`` and ``b_big`` (rows). No-op unless
        ``b_big > b_small`` - with one microbatch there is no second point.
        """
        if b_big <= b_small:
            return
        g_sq = (b_big * big_sq - b_small * small_sq) / (b_big - b_small)
        s = (small_sq - big_sq) / (1.0 / b_small - 1.0 / b_big)
        a = self.ema_alpha
        if self._g_sq_ema is None:
            self._g_sq_ema, self._s_ema = g_sq, s
        else:
            self._g_sq_ema = a * g_sq + (1 - a) * self._g_sq_ema
            self._s_ema = a * s + (1 - a) * self._s_ema
        self._updates += 1

    @property
    def ready(self) -> bool:
        return self._updates >= self.min_updates

    def noise_scale(self) -> Optional[float]:
        """``B_noise = S_ema / |G|^2_ema`` in rows, or None while unusable.

        Syncs (``float()`` on device tensors) - call on the decision cadence,
        not per step. None when the EMAs are cold or either is non-positive
        (a near-converged |G|^2 estimate can dip below zero; a decision made
        on that noise would be arbitrary).
        """
        if not self.ready or self._g_sq_ema is None:
            return None
        g = float(self._g_sq_ema)
        s = float(self._s_ema)
        if not (math.isfinite(g) and math.isfinite(s)) or g <= 0.0 or s <= 0.0:
            return None
        return s / g

    def internals(self) -> Dict[str, float]:
        """Current EMA values for telemetry (syncs); {} while cold."""
        if self._g_sq_ema is None:
            return {}
        return {
            "gov_signal_sq": float(self._g_sq_ema),
            "gov_noise_var": float(self._s_ema),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "g_sq_ema": None if self._g_sq_ema is None else float(self._g_sq_ema),
            "s_ema": None if self._s_ema is None else float(self._s_ema),
            "updates": self._updates,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        g, s = state.get("g_sq_ema"), state.get("s_ema")
        self._g_sq_ema = None if g is None else torch.tensor(float(g))
        self._s_ema = None if s is None else torch.tensor(float(s))
        self._updates = int(state.get("updates", 0))


class BatchTierController:
    """Doubling/halving accumulation tiers tracking the measured noise scale.

    The tier moves one step at a time and only when the measured B_noise sits
    more than ``deadband`` (in log2 units) away from the current effective
    batch. Note the geometry: moving one tier shifts the reference point by a
    full octave, so the NET hysteresis between a tier's up-threshold and the
    next tier's down-threshold is ``2*deadband - 1`` octaves. At 1.0 the
    up-threshold of tier k equals tier k+1's effective batch (move up only
    once B_noise has actually reached the next tier) and the EMA must swing a
    full 2x to flap; 0.75 left only a 1.41x window, which live estimates
    crossed routinely (observed 32<->64 flapping in abstractinator-f). The
    floor of 2 keeps the two-point estimator alive (one microbatch has no
    second measurement point); the ceiling is the configured
    target_batch_size.
    """

    deadband = 1.0  # log2 distance before a tier move; net hysteresis 2*db - 1

    def __init__(self, micro_batch: int, max_factor: int) -> None:
        self.micro_batch = max(1, int(micro_batch))
        self.min_factor = 2
        self.max_factor = max(self.min_factor, int(max_factor))

    def clamp(self, factor: int) -> int:
        return max(self.min_factor, min(self.max_factor, int(factor)))

    def desired_factor(self, current: int, noise_scale: Optional[float]) -> int:
        """Next accumulation factor given the measured noise scale (rows)."""
        current = self.clamp(current)
        if noise_scale is None or noise_scale <= 0.0:
            return current
        gap = math.log2(noise_scale / (current * self.micro_batch))
        if gap > self.deadband:
            return self.clamp(min(current * 2, self.max_factor))
        if gap < -self.deadband:
            return self.clamp(current // 2)
        return current


# ── Dashboard cards (universal, folded into get_metric_descriptions when a
#    governor is live - see praxis/metrics/descriptions.py) ─────────────────

_GOV_GROUP = "governor"

GOVERNOR_METRIC_DESCRIPTIONS: Dict[str, dict] = {
    "gov_noise_scale": {
        "description": (
            "Gradient noise scale B_noise = S/|G|^2 (rows), the measured "
            "critical batch size: below it gradient noise dominates and "
            "larger batches are near-free progress; above it the batch "
            "wastes data. Expected to grow as loss falls - the governor's "
            "effective batch should climb after it."
        ),
        "chart": {
            "title": "Batch Governor: Noise Scale vs Effective Batch",
            "y_label": "rows",
            "y_scale": "logarithmic",
            "group": _GOV_GROUP,
            "order": 10,
            "series_group": "gov_tracking",
            "series_label": "noise scale",
        },
    },
    "gov_effective_batch": {
        "description": (
            "Live effective batch (accumulation factor x microbatch rows). "
            "Moves in doubling tiers, one tier per decision, with a "
            "log2-deadband so estimator jitter can't flap it. Floor 2x "
            "microbatch (the estimator needs two points), ceiling "
            "target_batch_size."
        ),
        # No title/axis: rides gov_noise_scale's chart via series_group.
        "chart": {
            "group": _GOV_GROUP,
            "order": 11,
            "series_group": "gov_tracking",
            "series_label": "effective batch",
        },
    },
    "gov_next_val_batch": {
        "description": (
            "Raw-batch index the governor has pointed Lightning's validation "
            "trigger at - the next val_every optimizer-step boundary. "
            "Validation fires when the raw-batch count reaches this line."
        ),
        "chart": {
            "title": "Batch Governor: Validation Target",
            "y_label": "raw batches",
            "y_scale": "linear",
            "group": _GOV_GROUP,
            "order": 30,
            "series_group": "gov_val_cadence",
            "series_label": "next val target",
        },
    },
    "gov_raw_batches": {
        "description": (
            "Raw dataloader batches completed this epoch. When this line "
            "meets the validation target, validation runs; a target that "
            "gets crossed without a validation is a silent miss, visible "
            "here immediately."
        ),
        # No title/axis: rides gov_next_val_batch's chart via series_group.
        "chart": {
            "group": _GOV_GROUP,
            "order": 31,
            "series_group": "gov_val_cadence",
            "series_label": "raw batches",
        },
    },
    "gov_signal_sq": {
        "description": (
            "EMA of the unbiased |G|^2 estimate - squared norm of the true "
            "gradient. Falls as training converges; its decline is what "
            "drives the noise scale (and the batch) up."
        ),
        "chart": {
            "title": "Batch Governor: Estimator Internals",
            "y_label": "value",
            "y_scale": "logarithmic",
            "group": _GOV_GROUP,
            "order": 20,
            "series_group": "gov_internals",
            "series_label": "|G|² (signal)",
        },
    },
    "gov_noise_var": {
        "description": (
            "EMA of the unbiased S estimate - trace of the per-row gradient "
            "covariance. The numerator of the noise scale."
        ),
        # No title/axis: rides gov_signal_sq's chart via series_group.
        "chart": {
            "group": _GOV_GROUP,
            "order": 21,
            "series_group": "gov_internals",
            "series_label": "S (noise)",
        },
    },
}
