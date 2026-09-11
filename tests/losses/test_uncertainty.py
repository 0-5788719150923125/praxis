"""UncertaintyWeighting: Kendall-style learned log-variance per named objective, exp(-s) L + s."""

import pytest
import torch


def test_a_hard_objective_down_weights_itself():
    """s settles at log L, so the weight settles at 1/L: an objective stuck
    high contributes gradient scaled DOWN, and cannot drown a task that is
    already working."""
    from praxis.losses.uncertainty import UncertaintyWeighting

    uw = UncertaintyWeighting(("hard", "easy"))
    opt = torch.optim.SGD(uw.parameters(), lr=0.5)
    for _ in range(200):
        opt.zero_grad()
        (uw("hard", torch.tensor(16.0)) + uw("easy", torch.tensor(0.5))).backward()
        opt.step()
    w = uw.weights()
    assert w["hard"] == pytest.approx(1 / 16.0, rel=0.15)
    assert w["easy"] == pytest.approx(1 / 0.5, rel=0.15)
    assert w["hard"] < w["easy"]


def test_the_balance_cannot_mute_an_objective():
    """Without the `+ s` term the optimum is s -> inf and every auxiliary is
    silently deleted."""
    from praxis.losses.uncertainty import UncertaintyWeighting

    uw = UncertaintyWeighting(("x",))
    opt = torch.optim.SGD(uw.parameters(), lr=0.1)
    for _ in range(300):
        opt.zero_grad()
        uw("x", torch.tensor(3.0)).backward()
        opt.step()
    assert uw.weights()["x"] > 1e-3
