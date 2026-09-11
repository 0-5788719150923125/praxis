"""ArcHoPE: HoPE whose rotary phase is a learned per-band warp over position and
recurrent depth, identical to HoPE until the warp grows."""

import torch

from praxis import PraxisConfig
from praxis.encoding.archope import ArcHoPE
from praxis.encoding.hope import HoPE


def _pair():
    config = PraxisConfig(
        hidden_size=64, num_heads=4, num_queries=1, block_size=512, dropout=0.0
    )
    torch.manual_seed(0)
    return ArcHoPE(config), HoPE(config)


def _rotate(enc, x, depth=0):
    q, k, _ = enc.before_scores(x, x, x, current_depth=depth)
    return q, k


def test_warp_is_sized_to_the_kept_bands():
    arc, _ = _pair()
    assert 0 < arc._pos_dim <= 16  # head_dim
    bands = arc._pos_dim // 2
    for name in ("alpha", "beta", "gamma", "lambda", "rho"):
        assert getattr(arc, f"warp_{name}").shape == (bands,), name


def test_is_hope_at_init_at_every_depth():
    arc, hope = _pair()
    x = torch.randn(2, 4, 12, 16)
    for depth in (0, 3):
        for a, h in zip(_rotate(arc, x, depth), _rotate(hope, x, depth)):
            torch.testing.assert_close(a, h)


def test_the_warp_moves_the_phase_and_couples_depth():
    arc, _ = _pair()
    x = torch.randn(2, 4, 12, 16)
    at_init = _rotate(arc, x)[0]
    with torch.no_grad():
        arc.warp_lambda.fill_(0.5)
    warped = _rotate(arc, x)[0]
    assert not torch.allclose(warped, at_init)
    with torch.no_grad():
        arc.warp_rho.fill_(1.0)
    assert not torch.allclose(_rotate(arc, x, depth=2)[0], _rotate(arc, x, depth=0)[0])
