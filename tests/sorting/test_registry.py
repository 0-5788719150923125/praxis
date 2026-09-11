"""Every entry of the ``sorting`` registry: shape, exact sort order for the entries
that permute, and gradient flow."""

from types import SimpleNamespace

import pytest
import torch

from praxis import registry

SORTING = sorted(registry.namespace("sorting"))


def _sorter(key, width, ascending=False, **overrides):
    # hidden_size must match the feature dim: the positional-bias fields in this
    # registry allocate [hidden_size] parameters and fall back to identity when the
    # width disagrees, which would quietly stop exercising them.
    fields = {"sinkhorn_temperature": 0.1, "sinkhorn_iterations": 5, **overrides}
    config = SimpleNamespace(hidden_size=width, sort_ascending=ascending, **fields)
    return registry.lookup("sorting", key)(config)


@pytest.mark.parametrize("seq_len", [1, 3])
@pytest.mark.parametrize("key", SORTING)
def test_preserves_shape(key, seq_len):
    x = torch.randn(2, seq_len, 4)
    assert _sorter(key, 4)(x).shape == x.shape


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("key", SORTING)
def test_permuting_entries_return_the_sorted_values(key, ascending):
    """The forward is the exact sort, whatever the entry's backward does."""
    if not registry.lookup("sorting", key).permutes:
        pytest.skip(f"{key} does not permute its input")
    x = torch.randn(2, 3, 5)
    expected = torch.sort(x, dim=-1, descending=not ascending).values
    # A warm temperature makes the soft permutation far from hard: only a true
    # straight-through estimator still returns the sort.
    for tau in (0.001, 0.1, 10.0):
        y = _sorter(key, 5, ascending, sinkhorn_temperature=tau)(x)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)


@pytest.mark.parametrize("key", SORTING)
def test_gradient_flow(key):
    """Each input feeds exactly one output (or a convex blend summing to one), so
    the gradient of ``y.sum()`` sums to the element count for every entry."""
    x = torch.randn(2, 3, 4, requires_grad=True)
    _sorter(key, 4)(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad.sum(), torch.tensor(float(x.numel())))


def test_sinkhorn_gradient_follows_the_soft_permutation():
    """At a warm temperature the gradient of the top output spreads over every
    input, where a hard permutation would send it to the argmax alone."""
    x = torch.randn(1, 1, 5, requires_grad=True)
    y = _sorter("sinkhorn", 5, sinkhorn_temperature=10.0)(x)
    y[..., 0].sum().backward()
    assert (x.grad.abs() > 0).all()
    assert x.grad.flatten().argmax() == x.detach().flatten().argmax()
