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

Selected by name via the ``mixing`` registry (mirrors the project's other
registries); the chosen ``--orchestration-type`` profile names one. New variants
are registry entries, not new CLI knobs.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable, Optional

import torch
from torch import Tensor

from praxis import registry
from praxis.registry import Entry

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


registry.declare(
    "mixing",
    title="Expert mixing",
    doc=(
        "How a remote-expert pool combines its members at inference. Named by the "
        "chosen orchestration profile, not by a flag of its own. Values are "
        "zero-argument factories returning a mixer, which reduces stacked expert "
        "outputs ``[E, ...]`` to ``[...]``, so a selector can tune a variant without "
        "new flags (the same shape as the ``wrappers`` registry)."
    ),
    entries={
        "mean": Entry(
            lambda: _mean,
            "Pool average: the consensus, pure bias. Robust and deterministic.",
        ),
        "vote": Entry(
            lambda: _vote,
            (
                "Average the per-expert distributions (the CALM expert vote): softmax "
                "over the last dim, mean across experts, returned as log-probs so the "
                "result stays in logit space."
            ),
        ),
        "sample": Entry(
            lambda: partial(_sample, keep=0.5),
            (
                "Keep a random half of the experts (at least one), then average. The "
                "expected mix is the mean; the realized mix orbits it, which is the "
                "point - exploration off consensus at inference."
            ),
        ),
        "sample_quarter": Entry(
            lambda: partial(_sample, keep=0.25),
            "sample keeping a random quarter of the experts - higher variance.",
        ),
        "wave": Entry(
            lambda: partial(_wave, freq=1.0, phase=0.0),
            (
                "A standing wave over the expert index weights the pool: expert i gets "
                "``1 + cos(2*pi*i/E)``, normalized to sum 1, so peers compose by "
                "constructive and destructive interference rather than a flat mean. "
                "Deterministic in the peer ordering."
            ),
        ),
        "wave_high": Entry(
            lambda: partial(_wave, freq=2.0, phase=0.0),
            "wave at doubled frequency over the peer axis.",
        ),
    },
)


def build_mixer(name: str) -> Mixer:
    """Resolve a mixing-strategy name to a mixer callable."""
    if name not in registry.namespace("mixing"):
        raise KeyError(
            f"unknown mixing strategy {name!r}; choices: {sorted(registry.namespace("mixing"))}"
        )
    return registry.lookup("mixing", name)()
