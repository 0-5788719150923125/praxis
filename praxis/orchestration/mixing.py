"""Mixing strategies: combine many tiny experts' outputs into one.

Given a stack of per-expert outputs ``[E, ...]`` (E = number of experts that
answered), a mixer reduces them to a single output ``[...]``. The pool uses
these at inference; training is per-expert and detached, so it never mixes.

Three regimes from next/world_models.md, plus the standing-wave variant:

* ``mean``  - the consensus / pool average (pure bias, robust).
* ``vote``  - softmax-average of logits (the CALM expert-vote approach).
* ``sample``- stochastic: keep a random subset, then average. Expected value is
  the mean, but the variance is the point - excursions off consensus.
* ``wave``  - a standing wave over the expert index weights the pool: each
  expert gets a fixed-phase sinusoidal weight, so the mix is a learned-free
  interference pattern across peers rather than a flat average. The wave is
  deterministic in the expert ordering, so peers compose by constructive /
  destructive interference (the harmonic idea, applied to the peer axis).

Selected by name via ``MIXING_REGISTRY`` (mirrors the project's other
registries); the chosen ``--orchestration-type`` profile names one. New variants
are registry entries, not new CLI knobs.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable, Optional

import torch
from torch import Tensor

# A mixer maps stacked expert outputs [E, ...] -> [...].
Mixer = Callable[[Tensor], Tensor]


def _mean(outputs: Tensor) -> Tensor:
    return outputs.mean(dim=0)


def _vote(outputs: Tensor) -> Tensor:
    # Treat the last dim as logits: average the per-expert distributions, then
    # return log-probs so the result stays in logit space for downstream use.
    probs = torch.softmax(outputs, dim=-1).mean(dim=0)
    return torch.log(probs.clamp_min(1e-12))


def _sample(outputs: Tensor, *, keep: float = 0.5, generator=None) -> Tensor:
    # Keep a random subset of experts (at least one), then average. The expected
    # mix is the mean; the realized mix orbits it - exploration at inference.
    e = outputs.shape[0]
    k = max(1, int(round(e * keep)))
    if k >= e:
        return outputs.mean(dim=0)
    idx = torch.randperm(e, generator=generator, device=outputs.device)[:k]
    return outputs[idx].mean(dim=0)


def _wave(outputs: Tensor, *, freq: float = 1.0, phase: float = 0.0) -> Tensor:
    # A standing wave over the expert index: expert i gets weight
    # w_i = 1 + cos(2*pi*freq*i/E + phase), normalized to sum 1. Deterministic in
    # the peer ordering, so peers compose by interference, not a flat mean. With
    # E=1 this is just that expert; as E grows the wave shapes the consensus.
    e = outputs.shape[0]
    i = torch.arange(e, device=outputs.device, dtype=outputs.dtype)
    w = 1.0 + torch.cos(2 * math.pi * freq * i / max(1, e) + phase)
    w = (w / w.sum().clamp_min(1e-12)).view(e, *([1] * (outputs.dim() - 1)))
    return (outputs * w).sum(dim=0)


# Registry of mixing strategies. Values are zero-arg-callable factories so a
# selector can tune a variant without new CLI flags (same shape as the optimizer
# WRAPPER_REGISTRY).
MIXING_REGISTRY: dict[str, Callable[[], Mixer]] = {
    "mean": lambda: _mean,
    "vote": lambda: _vote,
    "sample": lambda: partial(_sample, keep=0.5),
    "sample_quarter": lambda: partial(_sample, keep=0.25),
    "wave": lambda: partial(_wave, freq=1.0, phase=0.0),
    "wave_high": lambda: partial(_wave, freq=2.0, phase=0.0),
}

MIXING_DESCRIPTIONS: dict[str, str] = {
    "mean": "Pool average (consensus / pure bias). Robust, deterministic.",
    "vote": "Average the per-expert distributions (CALM expert vote).",
    "sample": "Keep a random half of experts then average - stochastic.",
    "sample_quarter": "Keep a random quarter then average - higher variance.",
    "wave": "Standing wave over the expert index - peers compose by interference.",
    "wave_high": "Standing wave, doubled frequency over the peer axis.",
}


def build_mixer(name: str) -> Mixer:
    """Resolve a mixing-strategy name to a mixer callable."""
    if name not in MIXING_REGISTRY:
        raise KeyError(
            f"unknown mixing strategy {name!r}; choices: {sorted(MIXING_REGISTRY)}"
        )
    return MIXING_REGISTRY[name]()
