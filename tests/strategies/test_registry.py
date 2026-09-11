"""Every registered strategy folds a list of loss terms into one differentiable scalar."""

import pytest
import torch

from praxis import registry


@pytest.mark.parametrize("name", list(registry.namespace("strategies").keys()))
def test_every_strategy_folds_to_a_differentiable_scalar(name):
    """Imbalanced terms (four orders of magnitude apart) fold to a finite scalar
    whose gradient reaches every term."""
    strategy = registry.lookup("strategies", name)()
    losses = [
        torch.tensor(0.01, requires_grad=True),
        torch.tensor(100.0, requires_grad=True),
    ]

    total = strategy(losses)

    assert torch.is_tensor(total) and total.dim() == 0
    assert torch.isfinite(total)
    total.backward()
    for loss in losses:
        assert loss.grad is not None and torch.isfinite(loss.grad)
