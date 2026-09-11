"""Sweeps over the ``wrappers`` registry (praxis/optimization/wrappers.py)."""

import pytest
import torch
import torch.nn as nn

from praxis import registry
from praxis.optimization.wrappers import SequentialWrapper, wrappers_disable_schedule

# Only the schedule-free family runs without an LR schedule.
SCHEDULE_FREE = {"schedule_free", "gated_schedule_free", "wave_schedule_free"}
WRAPPERS = sorted(registry.namespace("wrappers"))


def test_the_schedule_free_family_is_registered():
    assert SCHEDULE_FREE <= set(WRAPPERS)
    assert {"trac", "ortho", "lookahead", "half_lion", "low_rank_moment"} <= set(
        WRAPPERS
    )


@pytest.mark.parametrize("key", WRAPPERS)
def test_disable_schedule_truth_table(key):
    assert wrappers_disable_schedule([key]) is (key in SCHEDULE_FREE)


def test_disable_schedule_composes():
    assert wrappers_disable_schedule(["ortho", "gated_schedule_free"]) is True
    assert wrappers_disable_schedule([]) is False


@pytest.mark.parametrize("key", WRAPPERS)
def test_wrapping_and_train_eval_leave_weights_untouched_at_init(key):
    """Before any step there is no state: wrapping, and train()/eval() where a
    wrapper swaps deployed weights, must be no-ops on the parameters."""
    torch.manual_seed(0)
    model = nn.Linear(8, 4, bias=False)
    w0 = model.weight.detach().clone()
    opt = SequentialWrapper([key])(torch.optim.SGD(model.parameters(), lr=0.05))
    if hasattr(opt, "train") and hasattr(opt, "eval"):
        opt.train()
        opt.eval()
    assert torch.equal(model.weight, w0)
