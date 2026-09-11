"""Sweeps over every ``halting`` profile."""

import pytest
import torch

from praxis import registry


@pytest.mark.parametrize("key", sorted(registry.namespace("halting").keys() - {"none"}))
@pytest.mark.parametrize("depth", [6, 18])
def test_the_ramp_toward_multiple_steps_survives(pmf, key, depth):
    """The property that keeps the model from exiting at one loop constantly:
    halting at 1 is LESS likely than halting at 2. True of the paper's curve and
    it has to stay true of any replacement, or the prior stops teaching depth."""
    p = pmf(key, depth)
    assert p[0] < p[1]


def test_other_profiles_leave_the_loop_untouched(halting):
    """The three loop hooks are identities everywhere but kl_log_reinject."""
    x = torch.randn(2, 5, 16)
    for key in registry.namespace("halting").keys() - {"kl_log_reinject"}:
        h = halting(key, 6)
        assert h.initial_state(x) is x
        assert h.inject(x, 0) is x
        assert h.settle(x) is x
