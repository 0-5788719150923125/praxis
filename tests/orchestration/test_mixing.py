"""Smoke tests for praxis.orchestration: the remote-expert pooling layer.

Uses trivially small in-process experts (LocalExpert wrapping a plain Linear
block) so the pool's mechanics - capacity reporting, non-blocking detached
training, stochastic-sampled inference, and the mixing strategies - are
exercised without any transport or real model.
"""

import torch

from praxis.orchestration import build_mixer
from praxis.orchestration.mixing import _sample, _wave

VOCAB = 16


def test_mixers_shapes_and_weighting():
    outputs = torch.randn(5, 2, 6, VOCAB)
    assert build_mixer("mean")(outputs).shape == (2, 6, VOCAB)
    assert build_mixer("vote")(outputs).shape == (2, 6, VOCAB)
    assert _wave(outputs, freq=1.0).shape == (2, 6, VOCAB)
    # sample keeps >=1 expert and averages
    g = torch.Generator().manual_seed(1)
    assert _sample(outputs, keep=0.4, generator=g).shape == (2, 6, VOCAB)


def test_wave_single_expert_is_identity():
    # With one expert the standing wave must reduce to that expert's output.
    single = torch.randn(1, 4, VOCAB)
    out = _wave(single, freq=1.0, phase=0.0)
    assert torch.allclose(out, single.squeeze(0), atol=1e-5)
