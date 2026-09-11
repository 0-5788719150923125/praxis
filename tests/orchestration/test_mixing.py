"""Mixing strategies (praxis/orchestration/mixing.py), which declares the
``mixing`` registry namespace: each reduces ``[E, ...]`` expert outputs to one."""

import pytest
import torch

from praxis import registry
from praxis.orchestration import build_mixer
from praxis.orchestration.mixing import _sample, _wave

VOCAB = 16


@pytest.mark.parametrize("key", list(registry.namespace("mixing")))
def test_every_mixer_reduces_the_expert_axis(key):
    outputs = torch.randn(5, 2, 6, VOCAB)
    mixed = build_mixer(key)(outputs)
    assert mixed.shape == (2, 6, VOCAB)
    assert torch.isfinite(mixed).all()


def test_mean_mixer_is_the_mean_and_sample_keeps_a_subset():
    outputs = torch.randn(5, 2, 6, VOCAB)
    torch.testing.assert_close(build_mixer("mean")(outputs), outputs.mean(dim=0))
    g = torch.Generator().manual_seed(1)
    kept = _sample(outputs, keep=0.4, generator=g)
    assert kept.shape == (2, 6, VOCAB)
    assert not torch.allclose(kept, outputs.mean(dim=0)), "sample kept every expert"


def test_wave_single_expert_is_identity():
    # With one expert the standing wave must reduce to that expert's output.
    single = torch.randn(1, 4, VOCAB)
    out = _wave(single, freq=1.0, phase=0.0)
    assert torch.allclose(out, single.squeeze(0), atol=1e-5)
