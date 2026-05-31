"""DecayBiasSort: additive rank-1 positional bias with a tail-decay envelope."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from praxis.sorting import SORTING_REGISTRY
from praxis.sorting.decay import DecayBiasSort


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
    # Monotone decay: head perturbed most, tail least (near zero).
    assert torch.all(norms[:-1] >= norms[1:] - 1e-6)
    assert float(norms[0].detach()) > float(norms[-1].detach())
    # The bias is the same direction scaled by g(t) = 1 - t/T.
    T = 8
    g = 1.0 - torch.arange(T, dtype=x.dtype) / T
    torch.testing.assert_close(delta, g.unsqueeze(-1) * s.bias)


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


def test_dim_mismatch_is_a_safe_noop():
    s = _sorter(hidden_size=16)
    x = torch.randn(2, 4, 8)  # wrong feature dim
    assert torch.equal(s(x), x)
