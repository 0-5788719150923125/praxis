"""SequentialWrapper: folding registry wrappers onto a base optimizer in order."""

import torch
import torch.nn as nn

from praxis.optimization.wrappers import SequentialWrapper


def _quadratic_problem():
    # Minimize ||W x - y||^2 with a LEARNABLE target (y = X @ W_true), so the
    # optimum is ~0 and a real optimizer should drive the loss down sharply.
    torch.manual_seed(0)
    model = nn.Linear(8, 4, bias=False)
    X = torch.randn(64, 8)
    Y = X @ torch.randn(8, 4)
    return model, X, Y


def test_sequential_wrapper_nests_in_order():
    model, _, _ = _quadratic_problem()
    base = torch.optim.SGD(model.parameters(), lr=0.05)
    wrapped = SequentialWrapper(["ortho", "schedule_free"])(base)
    # Outermost is schedule-free; unwrapping .optimizer reaches the base.
    assert type(wrapped).__name__ == "ScheduleFreeWrapper"
    inner = wrapped.optimizer
    assert type(inner).__name__ == "OrthoGrad"
    assert inner.optimizer is base


def test_sequential_wrapper_rejects_unknown_key():
    try:
        SequentialWrapper(["nope"])
    except ValueError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown wrapper key")
