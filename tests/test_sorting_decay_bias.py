"""DecayBiasSort: additive rank-1 positional bias with an absolute-position envelope."""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from praxis.sorting import SORTING_REGISTRY
from praxis.sorting.decay import (
    TAU_INIT,
    TAU_MAX,
    TAU_MIN,
    DecayBiasSort,
    bounded_tau,
    tau_logit,
)


def _sorter(hidden_size=16):
    return DecayBiasSort(SimpleNamespace(hidden_size=hidden_size))


def test_registered():
    assert SORTING_REGISTRY.get("decay_bias") is DecayBiasSort


def test_identity_at_init():
    # Zero-init bias -> starts as a no-op (the "if it doesn't work, code was
    # minimal" property).
    s = _sorter()
    x = torch.randn(2, 6, 16)
    assert torch.equal(s(x), x)


def test_additive_bias_decays_toward_tail():
    s = _sorter()
    with torch.no_grad():
        s.bias.copy_(torch.randn(16))
    x = torch.randn(1, 8, 16)
    out = s(x)
    delta = (out - x)[0]  # [T, H] - the applied bias per position
    norms = delta.norm(dim=-1)  # per-position bias magnitude
    # Monotone decay: head perturbed most, tail least.
    assert torch.all(norms[:-1] >= norms[1:] - 1e-6)
    assert float(norms[0].detach()) > float(norms[-1].detach())
    # The bias is the same direction scaled by g(t) = exp(-t/tau).
    t = torch.arange(8, dtype=torch.float32)
    g = torch.exp(-t / TAU_INIT)
    torch.testing.assert_close(delta, g.unsqueeze(-1) * s.bias)


def test_field_is_independent_of_sequence_length():
    """The regression this parameterization exists for.

    Under the old ``g(t) = 1 - t/T`` envelope, a token's bias depended on how
    long the sequence it was batched into happened to be. The same position must
    now land on the same field value at every length.
    """
    s = _sorter()
    with torch.no_grad():
        s.bias.copy_(torch.randn(16))
    x = torch.randn(1, 64, 16)
    full = s(x) - x
    for length in (4, 9, 17, 33, 64):
        clipped = s(x[:, :length]) - x[:, :length]
        torch.testing.assert_close(clipped, full[:, :length])


def test_offset_continues_the_field_for_cached_decode():
    """Cached decode feeds only the new suffix; ``offset`` has to continue it."""
    s = _sorter()
    with torch.no_grad():
        s.bias.copy_(torch.randn(16))
    x = torch.randn(1, 12, 16)
    full = s(x) - x
    # Feed the tail alone, telling the module where it actually sits.
    suffix = s(x[:, 8:], offset=8) - x[:, 8:]
    torch.testing.assert_close(suffix, full[:, 8:])
    # ...and without the offset it would wrongly restart at position 0.
    assert not torch.allclose(s(x[:, 8:]) - x[:, 8:], full[:, 8:], atol=1e-4)


def test_survives_layernorm_direction_change():
    # An additive per-feature bias changes direction, so normalization does NOT
    # erase it (the whole point vs a scalar amplitude scale).
    s = _sorter()
    with torch.no_grad():
        s.bias.copy_(torch.randn(16))
    ln = nn.LayerNorm(16)
    x = torch.randn(1, 5, 16)
    assert not torch.allclose(ln(s(x)), ln(x), atol=1e-5)


def test_bias_is_a_trainable_parameter():
    s = _sorter()
    names = {n for n, _ in s.named_parameters()}
    assert "bias" in names and s.bias.requires_grad
    # Gradient reaches the bias (so the optimizer can shape it).
    (s(torch.randn(2, 4, 16)).sum()).backward()
    assert s.bias.grad is not None and s.bias.grad.abs().sum() >= 0.0


def test_tau_is_learnable_and_bounded():
    s = _sorter()
    assert "log_tau" in {n for n, _ in s.named_parameters()}
    # Inits exactly on TAU_INIT, and stays inside the bounds at any extreme.
    assert math.isclose(float(bounded_tau(s.log_tau.detach())), TAU_INIT, rel_tol=1e-5)
    for z in (-1e4, -20.0, 0.0, 20.0, 1e4):
        tau = float(bounded_tau(torch.tensor([z])))
        assert TAU_MIN <= tau <= TAU_MAX
    assert math.isclose(tau_logit(TAU_INIT), float(s.log_tau.detach()), rel_tol=1e-6)


def test_tau_gradient_is_gated_by_the_bias():
    """tau only starts moving once the bias is nonzero - it has nothing to scale
    before that, so the module picks its horizon only after it wants one."""
    s = _sorter()
    s(torch.randn(2, 6, 16)).sum().backward()
    assert float(s.log_tau.grad.abs().sum()) == 0.0

    s = _sorter()
    with torch.no_grad():
        s.bias.copy_(torch.randn(16))
    s(torch.randn(2, 6, 16)).sum().backward()
    assert float(s.log_tau.grad.abs().sum()) > 0.0


def test_training_metrics_report_the_field():
    s = _sorter()
    metrics = s.training_metrics()
    assert metrics["sorting/bias_norm"] == 0.0  # identity at init
    assert math.isclose(metrics["sorting/decay_tau"], TAU_INIT, rel_tol=1e-5)
    with torch.no_grad():
        s.bias.copy_(torch.ones(16))
    assert math.isclose(s.training_metrics()["sorting/bias_norm"], 4.0, rel_tol=1e-5)


def test_dim_mismatch_is_a_safe_noop():
    s = _sorter(hidden_size=16)
    x = torch.randn(2, 4, 8)  # wrong feature dim
    assert torch.equal(s(x), x)
