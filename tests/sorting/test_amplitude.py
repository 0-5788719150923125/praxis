"""AmplitudeFieldSort's multiplicative per-feature modulation. The additive half it
inherits from DecayBiasSort is covered in test_decay.py."""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from praxis.sorting.amplitude import MAX_PERIOD, MIN_PERIOD, AmplitudeFieldSort


def _sorter(hidden_size=32):
    return AmplitudeFieldSort(SimpleNamespace(hidden_size=hidden_size))


def test_wavelength_spectrum_has_variety():
    s = _sorter()
    assert float(s.periods.min()) <= MIN_PERIOD + 1e-4
    assert float(s.periods.max()) >= MAX_PERIOD - 1e-2
    assert float(s.periods.max() / s.periods.min()) > 10.0  # genuinely spread


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


def test_training_metrics_report_the_modulation_depth():
    s = _sorter()
    assert s.training_metrics()["sorting/mod_depth"] == 0.0  # identity at init
    with torch.no_grad():
        s.amp.copy_(torch.full((32,), 1.0))
    assert math.isclose(
        s.training_metrics()["sorting/mod_depth"], math.tanh(1.0), rel_tol=1e-5
    )
