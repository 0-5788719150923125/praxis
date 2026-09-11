"""DecayBiasSort: additive rank-1 positional bias with an absolute-position envelope.

AmplitudeFieldSort extends it with a multiplicative per-feature modulation, so the
behaviors the two share are checked here on both classes; the modulation itself is
covered in test_amplitude.py.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from praxis.sorting.amplitude import AmplitudeFieldSort
from praxis.sorting.decay import (
    TAU_INIT,
    TAU_MAX,
    TAU_MIN,
    DecayBiasSort,
    bounded_tau,
    tau_logit,
)

WIDTH = 16

both = pytest.mark.parametrize("cls", [DecayBiasSort, AmplitudeFieldSort])


def _sorter(cls=DecayBiasSort, hidden_size=WIDTH):
    return cls(SimpleNamespace(hidden_size=hidden_size))


def _shaped(cls):
    """A sorter moved off its zero init, both halves of the field where it has two."""
    s = _sorter(cls)
    with torch.no_grad():
        s.bias.copy_(torch.randn(WIDTH))
        if hasattr(s, "amp"):
            s.amp.copy_(torch.randn(WIDTH))
    return s


@both
def test_identity_at_init(cls):
    # Zero-init bias (and amp) -> starts as a no-op.
    x = torch.randn(2, 6, WIDTH)
    torch.testing.assert_close(_sorter(cls)(x), x, rtol=0, atol=1e-6)


@both
def test_additive_bias_decays_toward_tail(cls):
    s = _sorter(cls)
    with torch.no_grad():
        s.bias.copy_(torch.randn(WIDTH))  # amp stays 0 -> only the additive part
    x = torch.randn(1, 8, WIDTH)
    delta = (s(x) - x)[0]  # [T, H] - the applied bias per position
    norms = delta.norm(dim=-1)
    # Monotone decay: head perturbed most, tail least.
    assert torch.all(norms[:-1] >= norms[1:] - 1e-5)
    assert float(norms[0].detach()) > float(norms[-1].detach())
    # The bias is the same direction scaled by g(t) = exp(-t/tau).
    t = torch.arange(8, dtype=torch.float32)
    torch.testing.assert_close(delta, torch.exp(-t / TAU_INIT).unsqueeze(-1) * s.bias)


@both
def test_field_is_independent_of_sequence_length(cls):
    """The same absolute position lands on the same field value at every length,
    so a token's bias does not depend on the length it was batched into."""
    s = _shaped(cls)
    x = torch.randn(1, 64, WIDTH)
    full = s(x)
    for length in (4, 9, 17, 33, 64):
        torch.testing.assert_close(s(x[:, :length]), full[:, :length])


@both
def test_offset_continues_the_field_for_cached_decode(cls):
    """Cached decode feeds only the new suffix; ``offset`` has to continue it."""
    s = _shaped(cls)
    x = torch.randn(1, 12, WIDTH)
    full = s(x)
    torch.testing.assert_close(s(x[:, 8:], offset=8), full[:, 8:])
    # A single-token step (the generation case) lands on its true position.
    torch.testing.assert_close(s(x[:, 11:], offset=11), full[:, 11:])
    # ...and without the offset it would wrongly restart at position 0.
    assert not torch.allclose(s(x[:, 8:]), full[:, 8:], atol=1e-4)


def test_survives_layernorm_direction_change():
    # An additive per-feature bias changes direction, so normalization does NOT
    # erase it (the whole point vs a scalar amplitude scale).
    s = _shaped(DecayBiasSort)
    ln = nn.LayerNorm(WIDTH)
    x = torch.randn(1, 5, WIDTH)
    assert not torch.allclose(ln(s(x)), ln(x), atol=1e-5)


@both
def test_field_parameters_are_trainable(cls):
    s = _sorter(cls)
    params = dict(s.named_parameters())
    names = {"bias", "log_tau"} | ({"amp"} if cls is AmplitudeFieldSort else set())
    assert names <= set(params)
    s(torch.randn(2, 4, WIDTH)).sum().backward()
    # log_tau is gated by the bias (test below); every other field parameter
    # gets gradient straight from the zero init.
    for name in names - {"log_tau"}:
        assert params[name].grad.abs().sum() > 0, name


def test_tau_is_learnable_and_bounded():
    s = _sorter()
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
    s(torch.randn(2, 6, WIDTH)).sum().backward()
    assert float(s.log_tau.grad.abs().sum()) == 0.0

    s = _shaped(DecayBiasSort)
    s(torch.randn(2, 6, WIDTH)).sum().backward()
    assert float(s.log_tau.grad.abs().sum()) > 0.0


@both
def test_training_metrics_report_the_field(cls):
    s = _sorter(cls)
    metrics = s.training_metrics()
    assert metrics["sorting/bias_norm"] == 0.0  # identity at init
    assert math.isclose(metrics["sorting/decay_tau"], TAU_INIT, rel_tol=1e-5)
    with torch.no_grad():
        s.bias.copy_(torch.ones(WIDTH))
    assert math.isclose(
        s.training_metrics()["sorting/bias_norm"], math.sqrt(WIDTH), rel_tol=1e-5
    )


@both
def test_dim_mismatch_is_a_safe_noop(cls):
    s = _sorter(cls)
    x = torch.randn(2, 4, WIDTH // 2)  # wrong feature dim
    assert torch.equal(s(x), x)
