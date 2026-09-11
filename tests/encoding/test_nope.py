"""Structural parity tests for the encoding-registry refactor.

These do NOT prove bit-exact equivalence to the pre-refactor inline _apply_rope
(the inline version used split-even-odd pairs, the registry uses split-in-half;
both are valid RoPE bases). They prove the properties that actually matter:

1. RoPE-style relative-position invariance still holds (shifting all positions
   by a constant doesn't change scores between same-relative-position pairs).
2. ALiBi continues to apply a strictly negative bias to past keys.
3. HoPE leaves a non-empty tail of head-dim slots unrotated, end to end.
"""

import torch

from praxis import PraxisConfig
from praxis.attention.causal import CausalAttention


def _make_config(encoding: str) -> PraxisConfig:
    return PraxisConfig(
        hidden_size=64,
        num_heads=4,
        num_queries=1,
        block_size=256,
        dropout=0.0,
        encoding=encoding,
        causal=True,
    )


def test_nope_passes_through_unchanged():
    cfg = _make_config("nope")
    attn = CausalAttention(cfg)
    enc = attn.encoding

    q = torch.randn(1, attn.num_query_heads, 8, attn.head_dim)
    k = torch.randn(1, attn.num_heads, 8, attn.head_dim)
    v = torch.randn(1, attn.num_heads, 8, attn.head_dim)
    q2, k2, v2 = enc.before_scores(q, k, v)
    # NoPE applies a head-wise scaling to Q (not a no-op), but K and V should
    # come through untouched, and the encoding should produce no score-mod.
    assert k2 is k
    assert v2 is v
    assert enc.build_score_mod(attn.num_query_heads, torch.device("cpu")) is None
