"""A registry-profile builder and the Monte-Carlo loop-count curves, shared by the
halting tests."""

import collections
from types import SimpleNamespace

import pytest
import torch

from praxis import registry

SAMPLES = 20_000

# One Monte-Carlo curve per distinct sampler, shared by every test in the
# directory: profiles that differ only outside the sampler (kl_log_reinject is
# kl_log's sampler and prior) read the same curve instead of paying ~3s again.
_CURVES = {}


def _build(key, depth, num_layers=1):
    return registry.lookup("halting", key)(
        SimpleNamespace(depth=depth, num_layers=num_layers, hidden_size=16)
    )


def _pmf(key, depth):
    """Monte-Carlo the real sampler. Deliberately not an analytic
    reimplementation - the point is to measure what training will see."""
    module = _build(key, depth)
    signature = (
        type(module)._sample_loop_count,
        module.r_bar,
        module.sigma,
        module.max_loops,
    )
    if signature not in _CURVES:
        torch.manual_seed(0)
        counts = collections.Counter(
            module._sample_loop_count() for _ in range(SAMPLES)
        )
        _CURVES[signature] = tuple(
            counts[r] / SAMPLES for r in range(1, module.max_loops + 1)
        )
    return _CURVES[signature]


@pytest.fixture
def halting():
    """``halting(key, depth, num_layers=1)`` builds a registry profile."""
    return _build


@pytest.fixture
def pmf():
    """``pmf(key, depth)``: the profile's loop-count distribution over 1..max_loops."""
    return _pmf
