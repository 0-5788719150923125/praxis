import math

import numpy as np
import torch
import torch.nn as nn

from .base import register_sorting
from .decay import DecayBiasSort

GOLDEN = (1.0 + 5.0**0.5) / 2.0  # irrational -> Weyl-equidistributed phases
MIN_PERIOD = 4.0  # fastest feature: one cycle every 4 positions
MAX_PERIOD = 4096.0  # slowest feature: one cycle every 4096 positions


@register_sorting("amplitude_field")
class AmplitudeFieldSort(DecayBiasSort):
    """Amplitude modulation coupled through BOTH axes (sequence and feature).

    Two complementary, norm-surviving components, identity at init:

    * additive positional decay bias (sequence): ``+ g(t)*v``,
      ``g(t) = exp(-t/tau)`` - inherited wholesale from ``decay_bias``.
    * multiplicative per-feature frequency modulation (feature x sequence):
      ``* (1 + tanh(a_d)*sin(2*pi*t/lambda_d + phi_d))``. Each feature ``d``
      oscillates over the sequence at its own wavelength ``lambda_d`` (a
      geometric spectrum from MIN_PERIOD to MAX_PERIOD *positions*), with
      frozen Weyl phases. The modulation is per-feature, so it changes
      direction (survives norm) and never zeroes the vector; ``tanh`` keeps
      the factor in ``(0, 2)``.

    Both axes are anchored to ABSOLUTE position, never to the sequence length -
    see ``DecayBiasSort`` for why. The wavelength spectrum is in positions
    rather than cycles-per-window for the same reason: cycles-per-window makes
    feature ``d``'s phase at a given token a function of how long that token's
    batch happened to be, so the per-feature gain fed to the head changes with
    batch shape. The spectrum stays FROZEN (and independent of any config knob,
    which would re-scramble the basis across configs); the learnable amplitudes
    are what select which bands matter.

    ``v``, the per-feature amplitudes ``a``, and the decay length ``tau`` are
    learnable; ``v`` and ``a`` are zero-init, so the module starts as identity
    and grows only if it helps. NB: on a head that already carries a harmonic
    field (prismatic), the multiplicative part overlaps that field - distinct
    value is clearest on non-harmonic heads or applied at the input side. Note
    also that a distance-based head (``CrystalClassifier``) is sensitive to
    ``||x||``, not only to direction, so the multiplicative part reaches its
    logits directly rather than being normalized away.
    """

    metric_descriptions = {
        **DecayBiasSort.metric_descriptions,
        "sorting/mod_depth": {
            "description": (
                "Mean |tanh(a_d)| over features: the typical depth of the "
                "per-feature frequency modulation, i.e. the fraction by which "
                "a feature's gain swings above/below 1 across the sequence. "
                "Zero-init, so 0 means the multiplicative half is inert; 0.5 "
                "means a typical feature ranges over a 3x gain span."
            ),
            "chart": {
                "title": "Frequency Modulation Depth",
                "y_label": "mean |tanh(amp)|",
                "group": "sorting",
                "order": 30,
            },
        },
    }

    def __init__(self, config):
        super().__init__(config)
        h = int(config.hidden_size)
        self.amp = nn.Parameter(torch.zeros(h))  # per-feature mod depth (pre-tanh)

        d = np.arange(1, h + 1, dtype=np.float64)
        periods = np.geomspace(MIN_PERIOD, MAX_PERIOD, h)  # per-feature wavelengths
        phases = 2.0 * math.pi * (d * GOLDEN - np.floor(d * GOLDEN))  # Weyl phases
        self.register_buffer(
            "periods", torch.from_numpy(periods).float(), persistent=False
        )
        self.register_buffer(
            "phases", torch.from_numpy(phases).float(), persistent=False
        )

    def _modulation(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """``[T, H]`` per-feature gain in ``(0, 2)``, from absolute position."""
        periods = self.periods.to(device=t.device, dtype=t.dtype)
        phases = self.phases.to(device=t.device, dtype=t.dtype)
        # Wrap into one cycle before scaling: absolute position is unbounded, and
        # a raw 2*pi*t/lambda would bleed fp32 precision out of sin() on a long
        # context. sin is 2*pi-periodic, so this is the same field.
        cycles = torch.remainder(t.unsqueeze(-1) / periods, 1.0)  # [T, H]
        phase = 2.0 * math.pi * cycles + phases  # [T, H]
        mod = 1.0 + torch.tanh(self.amp.to(t.dtype)) * torch.sin(phase)
        return mod.to(dtype)

    def forward(
        self, hidden_states: torch.Tensor, offset: int = 0, **kwargs
    ) -> torch.Tensor:
        if not self._applies(hidden_states):
            return hidden_states
        t = self._positions(hidden_states, offset)  # [T] fp32, absolute
        out = self._add_bias(hidden_states, t)
        return out * self._modulation(t, hidden_states.dtype)

    def training_metrics(self) -> dict:
        metrics = super().training_metrics()
        with torch.no_grad():
            metrics["sorting/mod_depth"] = float(
                torch.tanh(self.amp.detach().float()).abs().mean()
            )
        return metrics
