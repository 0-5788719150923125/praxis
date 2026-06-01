import math

import numpy as np
import torch
import torch.nn as nn

from .base import NoSort, register_sorting

GOLDEN = (1.0 + 5.0**0.5) / 2.0  # irrational -> Weyl-equidistributed phases
MIN_CYCLES = 0.5  # slowest feature: half a cycle across the sequence
MAX_CYCLES = 16.0  # fastest feature: 16 cycles across the sequence


@register_sorting("amplitude_field")
class AmplitudeFieldSort(NoSort):
    """Amplitude modulation coupled through BOTH axes (sequence and feature).

    Two complementary, norm-surviving components, identity at init:

    * additive positional decay bias (sequence): ``+ g(t)*v``, ``g(t)=1-t/T`` -
      strong absolute-position shaping on the oldest context, ~0 at the tail
      (same as ``decay_bias``; additive, so a per-position norm can't divide it
      out, and the predictive tail keeps its content).
    * multiplicative per-feature frequency modulation (feature x sequence):
      ``* (1 + tanh(a_d)*sin(2*pi*f_d*t/T + phi_d))``. Each feature ``d``
      oscillates over the sequence at its own frequency ``f_d`` (a geometric
      spectrum from MIN_CYCLES to MAX_CYCLES), with frozen Weyl phases. The
      modulation is per-feature, so it changes direction (survives norm) and
      never zeroes the vector; ``tanh`` keeps the factor in ``(0, 2)``.

    ``v`` and the per-feature amplitudes ``a`` are learnable and zero-init, so
    the module starts as identity and grows only if it helps. The frequency
    spectrum and phases are frozen (no per-experiment tuning). NB: on a head
    that already carries a harmonic field (prismatic), the multiplicative part
    overlaps that field - distinct value is clearest on non-harmonic heads or
    applied at the input side.
    """

    def __init__(self, config):
        super().__init__(config)
        h = int(config.hidden_size)
        self.bias = nn.Parameter(torch.zeros(h))  # additive positional bias dir
        self.amp = nn.Parameter(torch.zeros(h))  # per-feature mod depth (pre-tanh)

        d = np.arange(1, h + 1, dtype=np.float64)
        freqs = np.geomspace(MIN_CYCLES, MAX_CYCLES, h)  # per-feature spectrum
        phases = 2.0 * math.pi * (d * GOLDEN - np.floor(d * GOLDEN))  # Weyl phases
        self.register_buffer("freqs", torch.from_numpy(freqs).float(), persistent=False)
        self.register_buffer(
            "phases", torch.from_numpy(phases).float(), persistent=False
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        seq_len = hidden_states.shape[-2]
        if seq_len == 0 or self.bias.shape[-1] != hidden_states.shape[-1]:
            return hidden_states
        device, dtype = hidden_states.device, hidden_states.dtype
        t = torch.arange(seq_len, device=device, dtype=dtype)

        # additive positional decay bias (sequence axis)
        g = 1.0 - t / seq_len  # [T]: 1 at head -> ~0 at tail
        out = hidden_states + g.unsqueeze(-1) * self.bias.to(dtype)

        # multiplicative per-feature frequency modulation (feature x sequence)
        t_norm = (t / seq_len).unsqueeze(-1)  # [T, 1] in [0, 1)
        freqs = self.freqs.to(device=device, dtype=dtype)
        phases = self.phases.to(device=device, dtype=dtype)
        phase = 2.0 * math.pi * freqs * t_norm + phases  # [T, H]
        mod = 1.0 + torch.tanh(self.amp.to(dtype)) * torch.sin(phase)  # [T, H] in (0,2)
        return out * mod
