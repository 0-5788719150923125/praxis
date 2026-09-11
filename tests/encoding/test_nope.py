"""NoPE: no positional signal, so keys come through untouched."""

import torch

from praxis import PraxisConfig
from praxis.encoding.nope import NoPE


def test_keys_pass_through_unchanged():
    # NoPE scales Q head-wise (not a no-op), but K must come through as-is.
    enc = NoPE(PraxisConfig(hidden_size=64, num_heads=4, num_queries=1))
    q, k, v = (torch.randn(1, 4, 8, 16) for _ in range(3))
    _, k2, _ = enc.before_scores(q, k, v)
    assert k2 is k
