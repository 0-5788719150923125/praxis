"""Tests for the build_score_mod hook on the encoding registry.

NoPE/RoPE/HoPE must return None (they work via before_scores); ALiBi must
return a closure that matches the inline alibi_score_mod previously
hand-rolled in CausalAttention / InfiniAttention.
"""

import torch

from praxis import PraxisConfig
from praxis.encoding import ENCODING_REGISTRY


def _cfg(encoding: str) -> PraxisConfig:
    return PraxisConfig(
        hidden_size=64,
        num_heads=4,
        num_queries=1,
        block_size=512,
        dropout=0.0,
        encoding=encoding,
    )


def test_nope_rope_hope_return_none():
    for name in ("nope", "rope", "hope"):
        enc = ENCODING_REGISTRY[name](_cfg(name))
        mod = enc.build_score_mod(num_heads=4, device=torch.device("cpu"))
        assert mod is None, f"{name} should return None from build_score_mod"


def test_alibi_no_ghost_matches_simple_bias():
    enc = ENCODING_REGISTRY["alibi"](_cfg("alibi"))
    device = torch.device("cpu")
    mod = enc.build_score_mod(num_heads=4, device=device, ghost_offset=0)
    slopes = enc.compute_slopes(4, device)
    # Spot-check a few (h, q, kv) tuples.
    for h, q_idx, kv_idx in [(0, 0, 0), (1, 5, 3), (3, 10, 12)]:
        score = torch.tensor(0.7)
        expected = score + slopes[h] * (kv_idx - q_idx)
        got = mod(
            score, b=0, h=h, q_idx=torch.tensor(q_idx), kv_idx=torch.tensor(kv_idx)
        )
        torch.testing.assert_close(got, expected)


def test_alibi_ghost_offset_matches_inline_closure():
    # Reproduces the inline alibi_score_mod from causal.py/infini.py:
    #   is_not_ghost = (kv_idx > 0).float()
    #   actual_kv = kv_idx - 1
    #   bias = slopes[h] * (actual_kv - q_idx) * is_not_ghost
    enc = ENCODING_REGISTRY["alibi"](_cfg("alibi"))
    device = torch.device("cpu")
    mod = enc.build_score_mod(num_heads=4, device=device, ghost_offset=1)
    slopes = enc.compute_slopes(4, device)
    for h, q_idx, kv_idx in [
        (0, 0, 0),  # ghost column -> bias should vanish
        (2, 4, 1),  # first real key, actual_kv=0
        (3, 7, 9),  # actual_kv=8
    ]:
        score = torch.tensor(0.3)
        q_t = torch.tensor(q_idx)
        kv_t = torch.tensor(kv_idx)
        is_not_ghost = (kv_t > 0).float()
        expected = score + slopes[h] * ((kv_t - 1) - q_t) * is_not_ghost
        got = mod(score, b=0, h=h, q_idx=q_t, kv_idx=kv_t)
        torch.testing.assert_close(got, expected)
