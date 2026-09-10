"""Make the harmonic field's partials beat, as far as the task will pay for it.

WHAT DISSONANCE IS HERE. Roughness between partials, in the Plomp-Levelt sense:
two components close enough in frequency to fall inside one critical band, but
not close enough to fuse, beat against each other. It is a property of WHICH
frequencies coexist, not of how many - a sawtooth carries energy in every
harmonic and is a consonant tone. So this term is not a spread penalty. It
weights every pair of temporal modes by how much they beat and asks the field
for more of it.

WHAT IT PUSHES AGAINST. The field's standing spectrum is concentrated at low
temporal frequency twice over: the fixed basis carries a radial 1/f decay, and
the smoothness prior holds low-frequency mass through a dual variable of its own
(``praxis/heads/harmonic.py``). Nothing has ever pushed the other way. Roughness
is the exact counterweight - adjacent modes only fall inside a critical band
partway UP the frequency axis - so the two duals pull in opposite directions and
settle wherever the language-modelling task lets them.

THE KERNEL. ``R(f_i, f_j) = exp(-3.5 x) - exp(-5.75 x)`` with
``x = |f_i - f_j| / (0.2 * min(f_i, f_j))``: the two-exponential fit to the
Plomp-Levelt dissonance curve, peaking at 0.22 critical bands. The five
constants are that published fit, not settings. Critical bandwidth is taken as
a fixed FRACTION of frequency because these modes have no absolute scale - they
are cycles per period, not hertz - and a constant fraction is the only
anchor-free choice. Consequence, computed rather than assumed: with the field's
integer harmonics the roughest pair sits near ``f_t ~ 23`` and the peak becomes
unreachable below ``F_t ~ 24``, where the term degenerates into "push mass
upward". The metric says which regime a run is in.

Only the TEMPORAL axis is scored. Beating is a phenomenon in time; ``f_d`` is a
frequency over the feature axis, where the notion has no meaning. Energy is
summed over ``f_d`` first.

THE BALANCE IS A DUAL VARIABLE, NOT A WEIGHT. A fixed coefficient on "be
rougher" is the experiment rather than a setting, so the strength is a
multiplier moved by whether the task is still improving: a fast and a slow EMA
of the main loss, ratcheting UP while the fast one leads and falling back FASTER
when it does not. The equilibrium is where roughness costs as much progress as
it buys, which is the balance the term exists to find. Asymmetric on purpose - a
slow climb and a quick retreat mean a run that starts to break pulls the term
off itself. The step counts microbatches rather than optimizer steps, so under
gradient accumulation the ratchet moves that much faster - the same convention
the field's own smoothness dual uses.

READ IT HONESTLY. Loss progress is confounded: a learning-rate schedule, the
batch governor and the data mix all move it, and none of them know about this
term. So the multiplier climbing is NOT evidence the roughness is helping. The
falsifier is the multiplier pinning at its cap early - the constraint never
bound and this was a fixed weight after all. ``dissonance_probe`` is the same
measurement with no gradient, for reading the spectrum before pushing on it.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from praxis.losses.regularizer_base import BaseRegularizer

try:
    # This forward mutates buffers (the dual state and the loss EMAs) and reads
    # a module found by walking the head, neither of which belongs in a traced
    # graph. Same reasoning as praxis/losses/harmonic_kl.py.
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

# Dual step sizes on the log-multiplier. Up while the task is still improving,
# down four times as fast when it is not. Fixed and model-agnostic: they are a
# rate on a dimensionless signed signal, not a scale on any tensor.
DUAL_ETA_UP = 0.01
DUAL_ETA_DOWN = 0.04

# Where the log-multiplier starts: softplus(-5) ~ 0.007, so the term is
# effectively off at step 0 and has to earn its strength. Starting at rho = 0
# would hand it softplus(0) = 0.69 - most of the cap - before a single step of
# evidence.
RHO_INIT = -5.0

# Cap on the multiplier. Bounds the term's influence no matter how long the
# ratchet runs, and makes "pinned at the cap" a readable failure rather than a
# silent takeover.
LAMBDA_MAX = 1.0

# Horizons for the two loss EMAs the controller compares. ~10 and ~100 steps:
# far enough apart that the fast one leads on a real trend and not on one batch.
CE_FAST = 0.9
CE_SLOW = 0.99


def roughness_kernel(n_modes: int, device=None) -> Tensor:
    """``[n, n]`` pairwise roughness over integer modes ``1..n``, peak 1.

    Zero on the diagonal (a mode does not beat with itself) and normalised by
    the curve's own maximum, so the quadratic form below lands in [0, 1]
    whatever ``n`` is.
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


def _find_field(head) -> Optional[nn.Module]:
    """The harmonic field under ``head``, or None.

    Located by the method an objective on the spectrum needs
    (``amplitude_energy``) rather than by class, so this file imports nothing
    from ``praxis.heads`` and keeps working wherever the field is mounted -
    a parallel head's stem, a sequential stage, or a bare HarmonicHead.
    """
    if head is None or not hasattr(head, "modules"):
        return None
    for module in head.modules():
        if callable(getattr(module, "amplitude_energy", None)):
            return module
    return None


class Dissonance(BaseRegularizer):
    """Reward beating between the field's temporal modes, held by a dual."""

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
                "of the maximum the kernel allows. 0 is partials that never beat; 1 "
                "is all mass on the roughest pair."
            ),
            "chart": {
                "title": "Spectral Roughness",
                "y_label": "Share of max",
                "y_scale": "linear",
                "group": "dissonance",
                "order": 20,
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
                "The dual multiplier: climbs while the main loss improves, retreats "
                "four times as fast when it does not. Pinned at its cap means the "
                "constraint never bound."
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

    def __init__(self, pad_id: int = 0, observe_only: bool = False):
        super().__init__()
        self.pad_id = pad_id
        # Measurement without force, the same split contrastive_isotropy draws:
        # the spectrum readings are the only way to see whether pushing on it
        # helped, and they must not live only on the path that pushes.
        self.observe_only = observe_only
        # Dual state and the two loss EMAs. Persistent: a multiplier that reset
        # to zero on resume would restart the ratchet from scratch every time
        # the run is picked up. -1 marks "no observation yet".
        self.register_buffer("rho", torch.full((1,), RHO_INIT))
        self.register_buffer("ce_fast", torch.full((1,), -1.0))
        self.register_buffer("ce_slow", torch.full((1,), -1.0))
        # Built on the first forward, once F_t is known, and non-persistent:
        # it is a constant of the mode count, so a checkpoint carrying it would
        # only be a way to load a stale one.
        self._kernel: Optional[Tensor] = None
        # Slow copy of the spectrum, for the drift read only. Non-persistent
        # for the same reason harmonic_kl's teacher is: re-seeding from the live
        # spectrum on resume reads as zero drift for a while, which is a no-op,
        # where a stale saved copy would inject fictitious movement.
        self._ema_energy: Optional[Tensor] = None
        self._reported = False
        self._metrics: dict = {}

    def extra_repr(self) -> str:
        return "observe_only=True" if self.observe_only else ""

    def _lambda(self) -> float:
        """softplus(rho), capped. A float: it is a coefficient on the term, not
        something autograd should push around - only the constraint moves it."""
        if self.observe_only:
            return 0.0
        return float(
            torch.nn.functional.softplus(self.rho).clamp(max=LAMBDA_MAX).item()
        )

    @torch.no_grad()
    def _step_dual(self, main_loss: Optional[Tensor]) -> None:
        """One dual step from whether the task is still improving."""
        if self.observe_only or main_loss is None or not torch.is_tensor(main_loss):
            return
        value = main_loss.detach().float().reshape(-1)[0]
        if not torch.isfinite(value):
            return
        if float(self.ce_slow) < 0.0:  # first observation seeds both EMAs
            self.ce_fast.fill_(float(value))
            self.ce_slow.fill_(float(value))
            return
        self.ce_fast.mul_(CE_FAST).add_(value, alpha=1.0 - CE_FAST)
        self.ce_slow.mul_(CE_SLOW).add_(value, alpha=1.0 - CE_SLOW)
        improving = float(self.ce_fast) < float(self.ce_slow)
        step = DUAL_ETA_UP if improving else -DUAL_ETA_DOWN
        self.rho.add_(step).clamp_(-20.0, 20.0)

    def _roughness(self, p: Tensor) -> Tensor:
        """Share of the kernel's maximum roughness carried by ``p``.

        ``2 * p @ R @ p``: the pairwise sum counted once, scaled so that all
        mass on the single roughest pair reads exactly 1.
        """
        if self._kernel is None or self._kernel.shape[0] != p.shape[0]:
            self._kernel = roughness_kernel(p.shape[0], device=p.device)
        kernel = self._kernel.to(device=p.device, dtype=p.dtype)
        return 2.0 * (p @ (kernel @ p))

    @_no_compile
    def forward(self, hidden_states: Tensor, input_ids: Tensor, **ctx) -> Tensor:
        zero = hidden_states.new_zeros(())
        field = _find_field(ctx.get("head"))
        if field is None:
            if not self._reported:
                self._reported = True
                print(
                    "[dissonance] no harmonic field under this head; "
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

        self._step_dual(ctx.get("main_loss"))
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
                self._ema_energy.mul_(CE_SLOW).add_(detached, alpha=1.0 - CE_SLOW)
            self._metrics = {
                "dissonance": float(roughness.item()),
                "dissonance_centroid": float(((p * index).sum() / modes).item()),
                "dissonance_modes": float((1.0 / (p.pow(2).sum() * modes)).item()),
                "dissonance_lambda": lam,
                "dissonance_drift": float(drift.item()),
                "dissonance_loss": lam * float((1.0 - roughness).item()),
            }

        if self.observe_only or lam == 0.0:
            # An exact zero with no graph - a no-op in the sum, and nothing
            # downstream has to know this term is only watching.
            return zero
        # Non-negative and bounded by lambda, so the objective never rewards
        # itself with a negative term; minimising it maximises roughness.
        return lam * (1.0 - roughness)

    def training_metrics(self) -> dict:
        return dict(self._metrics)
