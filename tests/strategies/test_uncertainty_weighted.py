"""UncertaintyWeighted: homoscedastic-uncertainty weighting, 0.5/p^2 * L + log(1 + p^2) per term."""

import pytest
import torch

from praxis import registry


def test_params_materialize_on_the_first_call_and_learn():
    strategy = registry.lookup("strategies", "weighted")()
    losses = [torch.tensor(v, requires_grad=True) for v in (2.0, 3.0, 1.5)]

    total = strategy(losses)

    assert total.dim() == 0 and torch.isfinite(total)
    assert strategy.params.shape == torch.Size([3])
    total.backward()
    for loss in losses:
        assert loss.grad is not None
    assert strategy.params.grad is not None


@pytest.mark.parametrize(
    "name, coefficients",
    [
        # 0.5 / p^2 at p = [0.05, 0.5, 1, 30]
        ("weighted", [200.0, 2.0, 0.5, 0.5 / 900]),
        # p clamped to [0.1, 10] first: the small and large ends saturate
        ("weighted_clamped", [50.0, 2.0, 0.5, 0.005]),
    ],
)
def test_every_term_enters_with_a_positive_coefficient(name, coefficients):
    """Mixed-sign, extreme-magnitude terms: each loss keeps its sign in the
    total because its coefficient 0.5/p^2 is positive, and the clamped variant
    caps that coefficient at 50."""
    strategy = registry.lookup("strategies", name)()
    losses = [
        torch.tensor(v, requires_grad=True) for v in (100.0, -50.0, 1e-3, -1e-4)
    ]
    strategy(losses)  # materialize params
    with torch.no_grad():
        strategy.params.copy_(torch.tensor([0.05, 0.5, 1.0, 30.0]))

    total = strategy(losses)

    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(strategy.params.grad).all()
    for loss, expected in zip(losses, coefficients):
        assert loss.grad.item() > 0
        assert loss.grad.item() == pytest.approx(expected, rel=1e-5)
