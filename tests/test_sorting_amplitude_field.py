"""AmplitudeFieldSort: additive positional decay + per-feature freq modulation."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from praxis.sorting import SORTING_REGISTRY
from praxis.sorting.amplitude import MAX_CYCLES, MIN_CYCLES, AmplitudeFieldSort


def _sorter(hidden_size=32):
    return AmplitudeFieldSort(SimpleNamespace(hidden_size=hidden_size))


def test_registered():
    assert SORTING_REGISTRY.get("amplitude_field") is AmplitudeFieldSort


def test_identity_at_init():
    # bias and amp both zero-init -> starts as a no-op.
    s = _sorter()
    x = torch.randn(2, 7, 32)
    assert torch.allclose(s(x), x, atol=1e-6)


def test_frequency_spectrum_has_variety():
    s = _sorter(hidden_size=32)
    assert float(s.freqs.min()) <= MIN_CYCLES + 1e-4
    assert float(s.freqs.max()) >= MAX_CYCLES - 1e-4
    assert float(s.freqs.max() - s.freqs.min()) > 1.0  # genuinely spread


def test_additive_part_decays_toward_tail():
    s = _sorter()
    with torch.no_grad():
        s.bias.copy_(torch.randn(32))  # amp stays 0 -> only the additive part
    x = torch.randn(1, 8, 32)
    delta = (s(x) - x)[0]
    norms = delta.norm(dim=-1)
    assert torch.all(norms[:-1] >= norms[1:] - 1e-5)  # monotone decay to the tail


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


def test_params_trainable():
    s = _sorter()
    names = {n for n, _ in s.named_parameters()}
    assert {"bias", "amp"} <= names
    (s(torch.randn(2, 5, 32)).sum()).backward()
    assert s.bias.grad is not None and s.amp.grad is not None


def test_dim_mismatch_safe_noop():
    s = _sorter(hidden_size=32)
    x = torch.randn(2, 4, 16)
    assert torch.equal(s(x), x)
