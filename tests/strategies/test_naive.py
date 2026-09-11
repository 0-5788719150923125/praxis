"""NaiveSummation: the unweighted sum of the loss terms."""

import torch

from praxis import registry


def test_naive_strategy_is_the_plain_sum():
    strategy = registry.lookup("strategies", "naive")()
    losses = [torch.tensor(v, requires_grad=True) for v in (2.0, 3.0, 1.5)]

    total = strategy(losses)

    assert torch.isclose(total, torch.tensor(6.5))
    total.backward()
    for loss in losses:
        assert loss.grad == 1.0
