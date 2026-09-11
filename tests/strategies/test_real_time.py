"""RealTime: each term divided by its own detached value, so every term reads 1."""

import pytest
import torch

from praxis import registry


def test_every_term_is_normalised_to_one():
    strategy = registry.lookup("strategies", "real_time")()
    losses = [torch.tensor(v, requires_grad=True) for v in (2.0, 3.0, 1.5)]

    total = strategy(losses)

    assert total.dim() == 0
    assert total.item() == pytest.approx(3.0)
    total.backward()
    # d(L / L.detach()) / dL = 1 / L: gradient proportions follow the loss values.
    for loss in losses:
        assert loss.grad.item() == pytest.approx(1.0 / loss.item())


def test_a_zero_valued_term_is_skipped_rather_than_dividing_by_zero():
    strategy = registry.lookup("strategies", "real_time")()
    losses = [
        torch.tensor(2.0, requires_grad=True),
        torch.tensor(0.0, requires_grad=True),
        torch.tensor(1.5, requires_grad=True),
    ]

    total = strategy(losses)

    assert torch.isfinite(total)
    assert total.item() == pytest.approx(2.0)


@pytest.mark.xfail(
    strict=True,
    reason="real_time drops exactly-zero terms together with their gradient",
)
def test_a_zero_valued_surrogate_keeps_its_gradient():
    """prismatic9's ``arm_surgery`` is ``g - g.detach()``: value identically 0,
    gradient the only task signal the trunk gets. A fold that skips it by value
    discards that signal."""
    strategy = registry.lookup("strategies", "real_time")()
    h = torch.randn(4, requires_grad=True)
    other = torch.tensor(2.0, requires_grad=True)  # no path to h
    g = (h * 3.0).sum()
    surrogate = g - g.detach()

    strategy([other, surrogate]).backward()

    assert h.grad is not None and h.grad.abs().sum() > 0
