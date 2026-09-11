"""Keep the harmonic field at least as rough as the signal it multiplies.

Roughness is Plomp-Levelt dissonance between the field's temporal modes: two
partials close enough to share a critical band, but not to fuse, beat. It
depends on WHICH frequencies coexist, not how many, so it is not a spread
penalty. It is the counterweight to the smoothness prior
(``praxis/classifiers/harmonic.py``), since adjacent modes only beat partway up the
frequency axis.

The kernel is Sethares' fit to the Plomp-Levelt curve,
``R = exp(-3.5 x) - exp(-5.75 x)`` with ``x = |f_i - f_j| / (0.2 * min(f_i, f_j))``.
Critical bandwidth is a fixed fraction of frequency because the modes are cycles
per period, not hertz. Roughness is ``2 p'Kp`` over the normalized temporal
energy ``p``, divided by the grid's maximum (``roughness_ceiling``), so readings
lie in [0, 1].

The target is the same roughness measured on the hidden states the field
multiplies, as the smoothness prior sets its own target. The multiplier ascends
on ``target - roughness``, with ``rho`` clamped between ``RHO_INIT`` (effectively
off) and the value where softplus reaches the cap (anti-windup).

``dissonance_probe`` is the same measurement with no gradient and no dual step.
"""

import math
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from praxis.losses.regularizer_base import BaseRegularizer

try:
    # This forward mutates buffers (the dual state) and reads a module found by
    # walking the classifier, neither of which belongs in a traced graph. Same
    # reasoning as praxis/losses/harmonic_kl.py.
    from torch._dynamo import disable as _no_compile
except Exception:  # pragma: no cover

    def _no_compile(fn):
        return fn


# Plomp-Levelt roughness, in Sethares' two-exponential form. Published fit
# constants, not settings: the curve peaks at ln(b2/b1)/(b2-b1) = 0.22 critical
# bands and vanishes at both unison and wide intervals.
PL_B1 = 3.5
PL_B2 = 5.75

# Critical bandwidth as a fraction of frequency. These modes are cycles per
# period rather than hertz, so there is no absolute scale to anchor a hertz
# formula to; a constant fraction is the anchor-free choice.
CRITICAL_BAND = 0.2

# Dual ascent rate on rho, per microbatch. The plant is the same amplitude grid
# the smoothness dual steers, whose rate was measured over a 100x sweep of dual
# rate x grid-response time (praxis/classifiers/harmonic.py, SMOOTHNESS_DUAL_ETA).
DUAL_ETA = 0.003

# Where the log-multiplier starts, and the floor it returns to: softplus(-5) ~
# 0.007, so the term is effectively off until the constraint binds.
RHO_INIT = -5.0

# Cap on the multiplier, and the rho at which softplus reaches it. Clamping rho
# there, not above, is the anti-windup.
LAMBDA_MAX = 1.0
RHO_MAX = math.log(math.expm1(LAMBDA_MAX))

# The target is an EMA-smoothed setpoint read from a slice of the batch, as the
# smoothness prior's is: a couple of sequences keep it off the step time.
TARGET_EMA = 0.99
TARGET_PROBE_ROWS = 2

# Horizon of the slow spectrum copy the drift reading compares against.
DRIFT_EMA = 0.99


def roughness_kernel(n_modes: int, device=None) -> Tensor:
    """``[n, n]`` pairwise roughness over integer modes ``1..n``, peak 1.

    Zero on the diagonal (a mode does not beat with itself) and normalised by
    the curve's own maximum, so any single pair reads at most 1.
    """
    f = torch.arange(1, n_modes + 1, dtype=torch.float32, device=device)
    lo = torch.minimum(f.unsqueeze(0), f.unsqueeze(1))
    x = (f.unsqueeze(0) - f.unsqueeze(1)).abs() / (CRITICAL_BAND * lo)
    kernel = torch.exp(-PL_B1 * x) - torch.exp(-PL_B2 * x)
    peak_x = math.log(PL_B2 / PL_B1) / (PL_B2 - PL_B1)
    peak = math.exp(-PL_B1 * peak_x) - math.exp(-PL_B2 * peak_x)
    kernel = (kernel / peak).clamp_min(0.0)
    kernel.fill_diagonal_(0.0)
    return kernel


@lru_cache(maxsize=None)
def roughness_ceiling(n_modes: int) -> float:
    """Largest ``2 p'Kp`` over the probability simplex on ``n_modes`` modes.

    Replicator dynamics, ``p <- p * (Kp) / (p'Kp)``, never decreases ``p'Kp``
    for a symmetric non-negative kernel (Baum-Eagon), so each start climbs to a
    local maximum - but it can never grow a mode that starts at zero, so every
    start keeps a tenth of its mass spread over the whole grid. Started from the
    uniform spectrum and from contiguous bands of several widths across the
    grid, including its top edge, the best of them is the ceiling; for this
    kernel the maximizer is a band at the top of the grid.
    """
    if n_modes < 2:
        return 1.0
    kernel = roughness_kernel(n_modes).double()
    uniform = torch.full((n_modes,), 1.0 / n_modes, dtype=torch.float64)
    starts = [uniform]
    for width in (2, 4, 8, 12, 16, 24, 32, 48):
        if width > n_modes:
            break
        step = max(1, (n_modes - width) // 8)
        for lo in sorted(set(range(0, n_modes - width + 1, step)) | {n_modes - width}):
            band = torch.zeros(n_modes, dtype=torch.float64)
            band[lo : lo + width] = 1.0 / width
            starts.append(0.9 * band + 0.1 * uniform)
    p = torch.stack(starts)  # [S, n]
    for _ in range(3000):
        kp = p @ kernel
        value = (p * kp).sum(dim=-1, keepdim=True).clamp_min(1e-300)
        p = p * kp / value
    best = float((2.0 * (p * (p @ kernel)).sum(dim=-1)).max())
    return max(best, 1e-12)


def _find_field(classifier) -> Optional[nn.Module]:
    """The harmonic field under ``classifier``, or None.

    Located by the method an objective on the spectrum needs
    (``amplitude_energy``) rather than by class, so this file imports nothing
    from ``praxis.classifiers`` and keeps working wherever the field is mounted -
    a parallel classifier's stem, a sequential stage, or a bare
    HarmonicClassifier.
    """
    if classifier is None or not hasattr(classifier, "modules"):
        return None
    for module in classifier.modules():
        if callable(getattr(module, "amplitude_energy", None)):
            return module
    return None


class Dissonance(BaseRegularizer):
    """Hold the field's roughness at or above its input's, by a dual."""

    name = "dissonance"

    metric_descriptions = {
        "dissonance_loss": {
            "description": (
                "The penalty actually added to the objective, lambda * (1 - "
                "roughness). Sits in train_loss only - validation drops every aux "
                "term, so BPB is untouched."
            ),
            "chart": {
                "title": "Dissonance Penalty",
                "y_label": "Loss",
                "y_scale": "linear",
                "group": "dissonance",
                "group_order": 93,
                "order": 10,
            },
        },
        "dissonance": {
            "description": (
                "Plomp-Levelt roughness of the field's temporal spectrum, as a share "
                "of the most the kernel allows on this grid. 0 is partials that "
                "never beat; 1 is the roughest band."
            ),
            "chart": {
                "title": "Roughness vs Target",
                "y_label": "Share of max",
                "y_scale": "linear",
                "group": "dissonance",
                "order": 20,
                "series_group": "dissonance_pair",
                "series_label": "field",
            },
        },
        "dissonance_target": {
            "description": (
                "The same roughness measured on the hidden states the field "
                "multiplies - the constraint's target. The multiplier climbs while "
                "the field sits below it."
            ),
            "chart": {
                "title": "Roughness vs Target",
                "y_label": "Share of max",
                "y_scale": "linear",
                "group": "dissonance",
                "order": 21,
                "series_group": "dissonance_pair",
                "series_label": "target (signal)",
            },
        },
        "dissonance_centroid": {
            "description": (
                "Energy-weighted mean temporal mode, as a share of F_t. The axis the "
                "smoothness prior pulls down and this term pulls up - read the two "
                "together."
            ),
            "chart": {
                "title": "Spectral Centroid",
                "y_label": "Share of F_t",
                "y_scale": "linear",
                "group": "dissonance",
                "order": 30,
            },
        },
        "dissonance_modes": {
            "description": (
                "Share of temporal modes carrying the field - its participation "
                "ratio. Observed, never optimised: roughness can rise with or "
                "without spreading."
            ),
            "chart": {
                "title": "Active Temporal Modes",
                "y_label": "Share of F_t",
                "y_scale": "linear",
                "group": "dissonance",
                "order": 40,
            },
        },
        "dissonance_lambda": {
            "description": (
                "Dual multiplier: climbs while the field is smoother than its input, "
                "relaxes when rougher. At its softplus(-5) floor = not binding; at "
                "its cap = binding at full strength."
            ),
            "chart": {
                "title": "Dissonance Multiplier",
                "y_label": "Lambda",
                "y_scale": "linear",
                "group": "dissonance",
                "order": 50,
            },
        },
        "dissonance_drift": {
            "description": (
                "Relative L2 movement of the temporal spectrum against a slow EMA of "
                "itself - whether the standing field has stopped moving. Log-scaled."
            ),
            "chart": {
                "title": "Spectrum Drift",
                "y_label": "Relative L2",
                "y_scale": "logarithmic",
                "group": "dissonance",
                "order": 60,
            },
        },
    }

    # Buffers older checkpoints carry; dropped on load.
    _STALE_BUFFERS = ("seen", "ce_fast", "ce_slow", "gap_mean", "gap_var")

    def __init__(self, pad_id: int = 0, observe_only: bool = False):
        super().__init__()
        self.pad_id = pad_id
        # Measurement without force, the same split contrastive_isotropy draws:
        # the spectrum readings are the only way to see whether pushing on it
        # helped, and they must not live only on the path that pushes.
        self.observe_only = observe_only
        # Dual state, persistent so a resumed run picks the controller up where
        # it was. ``target`` < 0 means no observation yet.
        self.register_buffer("rho", torch.full((1,), RHO_INIT))
        self.register_buffer("target", torch.full((1,), -1.0))
        # Built on the first forward, once F_t is known, and non-persistent:
        # it is a constant of the mode count.
        self._kernel: Optional[Tensor] = None
        # Slow copy of the spectrum, for the drift read only. Non-persistent:
        # re-seeding on resume reads as zero drift for a while, which is a
        # no-op, where a stale saved copy would inject fictitious movement.
        self._ema_energy: Optional[Tensor] = None
        self._reported = False
        self._metrics: dict = {}

    def extra_repr(self) -> str:
        return "observe_only=True" if self.observe_only else ""

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        """Tolerate older checkpoints: drop stale buffers, start an absent target
        unobserved, and clamp rho into its range."""
        for name in self._STALE_BUFFERS:
            state_dict.pop(prefix + name, None)
        if prefix + "target" not in state_dict:
            state_dict[prefix + "target"] = torch.full((1,), -1.0)
        rho = state_dict.get(prefix + "rho")
        if rho is not None:
            state_dict[prefix + "rho"] = rho.clamp(RHO_INIT, RHO_MAX)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _lambda(self) -> float:
        """softplus(rho), capped. A float: it is a coefficient on the term, not
        something autograd should push around - only the constraint moves it."""
        if self.observe_only:
            return 0.0
        return float(
            torch.nn.functional.softplus(self.rho).clamp(max=LAMBDA_MAX).item()
        )

    def _roughness(self, p: Tensor) -> Tensor:
        """Share of the grid's maximum roughness carried by the distribution
        ``p`` over modes ``1..len(p)``."""
        n = p.shape[0]
        if self._kernel is None or self._kernel.shape[0] != n:
            self._kernel = roughness_kernel(n, device=p.device)
        kernel = self._kernel.to(device=p.device, dtype=p.dtype)
        return 2.0 * (p @ (kernel @ p)) / roughness_ceiling(n)

    @torch.no_grad()
    def _measure_target(self, hidden_states: Tensor, field: nn.Module, n: int):
        """Roughness of the field's input over the field's own modes, or None.

        The window is one field period (``field.T`` positions), so real-FFT bin
        ``f`` is the field's temporal mode ``f``. Rows shorter than a period
        have no such spectrum and are skipped.
        """
        period = getattr(field, "T", None)
        h = hidden_states
        if period is None or h.dim() != 3 or h.shape[-2] < period:
            return None
        h = h[:TARGET_PROBE_ROWS, :period].detach().float()
        centred = h - h.mean(dim=-2, keepdim=True)
        power = torch.fft.rfft(centred, dim=-2).abs().pow(2).sum(dim=(0, -1))
        power = power[1 : n + 1]
        total = power.sum()
        if power.numel() != n or not torch.isfinite(total) or total <= 0:
            return None
        return float(self._roughness(power / total))

    @torch.no_grad()
    def _step_dual(self, measured: Optional[float], roughness: float) -> None:
        """Fold the measured target into its EMA, then one dual-ascent step."""
        if not self.training:
            return
        if measured is not None and math.isfinite(measured):
            prev = float(self.target)
            self.target.fill_(
                measured
                if prev < 0.0
                else TARGET_EMA * prev + (1.0 - TARGET_EMA) * measured
            )
        if self.observe_only or float(self.target) < 0.0:
            return
        violation = float(self.target) - roughness
        if not math.isfinite(violation):
            return
        rho = float(self.rho) + DUAL_ETA * violation
        self.rho.fill_(min(max(rho, RHO_INIT), RHO_MAX))

    @_no_compile
    def forward(self, hidden_states: Tensor, input_ids: Tensor, **ctx) -> Tensor:
        zero = hidden_states.new_zeros(())
        field = _find_field(ctx.get("classifier"))
        if field is None:
            if not self._reported:
                self._reported = True
                print(
                    "[dissonance] no harmonic field under this classifier; "
                    "the term is inert for this run."
                )
            self._metrics = {}
            return zero

        # Sum over the feature-frequency axis: beating is temporal, and f_d is
        # not a time frequency.
        energy = field.amplitude_energy().sum(dim=-1)
        if energy.numel() < 2 or not torch.isfinite(energy).all():
            self._metrics = {}
            return zero

        p = energy / energy.sum().clamp_min(1e-12)
        roughness = self._roughness(p)
        self._step_dual(
            self._measure_target(hidden_states, field, p.numel()),
            float(roughness.detach()),
        )
        lam = self._lambda()

        with torch.no_grad():
            modes = p.numel()
            index = torch.arange(1, modes + 1, device=p.device, dtype=p.dtype)
            detached = energy.detach()
            if self._ema_energy is None or self._ema_energy.shape != detached.shape:
                self._ema_energy = detached.clone()
                drift = detached.new_zeros(())
            else:
                ref = self._ema_energy.norm().clamp_min(1e-12)
                drift = (detached - self._ema_energy).norm() / ref
                self._ema_energy.mul_(DRIFT_EMA).add_(detached, alpha=1.0 - DRIFT_EMA)
            self._metrics = {
                "dissonance": float(roughness.item()),
                "dissonance_centroid": float(((p * index).sum() / modes).item()),
                "dissonance_modes": float((1.0 / (p.pow(2).sum() * modes)).item()),
                "dissonance_lambda": lam,
                "dissonance_drift": float(drift.item()),
                "dissonance_loss": lam * float((1.0 - roughness).clamp(0.0, 1.0)),
            }
            if float(self.target) >= 0.0:
                self._metrics["dissonance_target"] = float(self.target)

        if self.observe_only or lam == 0.0:
            # An exact zero with no graph - a no-op in the sum, and nothing
            # downstream has to know this term is only watching.
            return zero
        # Bounded by [0, lambda]: minimising it raises roughness toward the
        # ceiling, and the clamp only guards float error at the ceiling itself.
        return lam * (1.0 - roughness.clamp(max=1.0))

    def training_metrics(self) -> dict:
        return dict(self._metrics)
