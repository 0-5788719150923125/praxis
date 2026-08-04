"""AmplitudeFieldSort: additive positional decay + per-feature freq modulation."""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from praxis.sorting import SORTING_REGISTRY
from praxis.sorting.amplitude import MAX_PERIOD, MIN_PERIOD, AmplitudeFieldSort
from praxis.sorting.decay import TAU_INIT, DecayBiasSort


def _sorter(hidden_size=32):
    return AmplitudeFieldSort(SimpleNamespace(hidden_size=hidden_size))


def test_registered():
    assert SORTING_REGISTRY.get("amplitude_field") is AmplitudeFieldSort


def test_extends_decay_bias():
    # amplitude_field IS decay_bias plus a feature modulation; sharing the class
    # keeps the two envelopes from drifting apart again.
    assert issubclass(AmplitudeFieldSort, DecayBiasSort)


def test_identity_at_init():
    # bias and amp both zero-init -> starts as a no-op.
    s = _sorter()
    x = torch.randn(2, 7, 32)
    assert torch.allclose(s(x), x, atol=1e-6)


def test_wavelength_spectrum_has_variety():
    s = _sorter(hidden_size=32)
    assert float(s.periods.min()) <= MIN_PERIOD + 1e-4
    assert float(s.periods.max()) >= MAX_PERIOD - 1e-2
    assert float(s.periods.max() / s.periods.min()) > 10.0  # genuinely spread


def test_additive_part_decays_toward_tail():
    s = _sorter()
    with torch.no_grad():
        s.bias.copy_(torch.randn(32))  # amp stays 0 -> only the additive part
    x = torch.randn(1, 8, 32)
    delta = (s(x) - x)[0]
    norms = delta.norm(dim=-1)
    assert torch.all(norms[:-1] >= norms[1:] - 1e-5)  # monotone decay to the tail
    t = torch.arange(8, dtype=torch.float32)
    torch.testing.assert_close(delta, torch.exp(-t / TAU_INIT).unsqueeze(-1) * s.bias)


def test_multiplicative_part_is_per_feature_and_survives_norm():
    s = _sorter()
    with torch.no_grad():
        s.amp.copy_(torch.randn(32))  # bias stays 0 -> only the multiplicative part
    x = torch.randn(1, 6, 32)
    out = s(x)
    # Per-feature modulation: the ratio out/x varies across features at a given
    # position (not a single scalar), so it changes direction.
    ratio = (out / x)[0].detach()  # [T, H]
    assert float(ratio[0].std()) > 1e-3  # features modulated differently
    # Direction change => a per-position norm does NOT erase it.
    ln = nn.LayerNorm(32)
    assert not torch.allclose(ln(out), ln(x), atol=1e-5)
    # Bounded factor keeps it stable (tanh -> (0, 2)).
    assert torch.isfinite(out).all()
    assert float(ratio.min()) > 0.0 and float(ratio.max()) < 2.0


def test_field_is_independent_of_sequence_length():
    """The regression this parameterization exists for.

    The old spectrum was cycles-per-window, so feature ``d``'s phase at a token
    was a function of the batch's padded length. Both halves of the field must
    now be a function of absolute position alone.
    """
    s = _sorter()
    with torch.no_grad():
        s.bias.copy_(torch.randn(32))
        s.amp.copy_(torch.randn(32))
    x = torch.randn(1, 64, 32)
    full = s(x)
    for length in (4, 9, 17, 33, 64):
        torch.testing.assert_close(s(x[:, :length]), full[:, :length])


def test_offset_continues_the_field_for_cached_decode():
    s = _sorter()
    with torch.no_grad():
        s.bias.copy_(torch.randn(32))
        s.amp.copy_(torch.randn(32))
    x = torch.randn(1, 12, 32)
    full = s(x)
    torch.testing.assert_close(s(x[:, 8:], offset=8), full[:, 8:])
    # A single-token step (the generation case) lands on its true position.
    torch.testing.assert_close(s(x[:, 11:], offset=11), full[:, 11:])
    assert not torch.allclose(s(x[:, 11:]), full[:, 11:], atol=1e-4)


def test_params_trainable():
    s = _sorter()
    names = {n for n, _ in s.named_parameters()}
    assert {"bias", "amp", "log_tau"} <= names
    (s(torch.randn(2, 5, 32)).sum()).backward()
    assert s.bias.grad is not None and s.amp.grad is not None


def test_training_metrics_report_both_halves():
    s = _sorter()
    metrics = s.training_metrics()
    assert metrics["sorting/bias_norm"] == 0.0
    assert metrics["sorting/mod_depth"] == 0.0  # identity at init
    assert math.isclose(metrics["sorting/decay_tau"], TAU_INIT, rel_tol=1e-5)
    with torch.no_grad():
        s.amp.copy_(torch.full((32,), 1.0))
    assert math.isclose(
        s.training_metrics()["sorting/mod_depth"], math.tanh(1.0), rel_tol=1e-5
    )


def test_dim_mismatch_safe_noop():
    s = _sorter(hidden_size=32)
    x = torch.randn(2, 4, 16)
    assert torch.equal(s(x), x)
