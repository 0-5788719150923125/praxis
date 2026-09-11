"""The training-time depth prior, which is the shape the halting signal learns.

`KLDivergenceHalting` samples a loop count per forward so the model never knows
how much compute it will get. The distribution those samples come from is the
experiment: it decides how much of the budget the model learns to treat as
routine, and the inference-time KL rule can only ever exit somewhere the prior
taught it to be useful.

What is pinned here is the SHAPE, not the sampler's internals - the ramp toward
multiple steps, and how fast the tail dies as the depth budget grows.
"""

import collections
import functools
from types import SimpleNamespace

import pytest
import torch

from praxis import registry

SAMPLES = 20_000


def _halting(key, depth, num_layers=1):
    return registry.lookup("halting", key)(
        SimpleNamespace(depth=depth, num_layers=num_layers, hidden_size=16)
    )


@functools.lru_cache(maxsize=None)
def _pmf(key, depth, seed=0):
    """Monte-Carlo the real sampler. Deliberately not an analytic reimplementation
    - the point is to measure what training will actually see. Cached because the
    sampler builds two distribution objects per draw and several tests read the
    same curve."""
    torch.manual_seed(seed)
    module = _halting(key, depth)
    counts = collections.Counter(module._sample_loop_count() for _ in range(SAMPLES))
    return tuple(counts[r] / SAMPLES for r in range(1, module.max_loops + 1))


@pytest.mark.parametrize("key", sorted(registry.namespace("halting").keys() - {"none"}))
@pytest.mark.parametrize("depth", [6, 18])
def test_the_ramp_toward_multiple_steps_survives(key, depth):
    """The property that keeps the model from exiting at one loop constantly:
    halting at 1 is LESS likely than halting at 2. True of the paper's curve and
    it has to stay true of any replacement, or the prior stops teaching depth."""
    p = _pmf(key, depth)
    assert p[0] < p[1]


def test_other_profiles_leave_the_loop_untouched():
    """The three loop hooks are identities everywhere but kl_log_reinject."""
    x = torch.randn(2, 5, 16)
    for key in registry.namespace("halting").keys() - {"kl_log_reinject"}:
        h = _halting(key, 6)
        assert h.initial_state(x) is x
        assert h.inject(x, 0) is x
        assert h.settle(x) is x
