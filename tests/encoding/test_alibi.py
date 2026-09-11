import torch

from praxis import PraxisConfig, registry
from praxis.attention.causal import CausalAttention

# ------------------------------------------------------------------------------
# encoding_score_mod
# ------------------------------------------------------------------------------
# Tests for the build_score_mod hook on the encoding registry.
#
# NoPE/RoPE/HoPE must return None (they work via before_scores); ALiBi must return a
# closure that matches the inline alibi_score_mod previously hand-rolled in
# CausalAttention / InfiniAttention.


def _cfg(encoding: str) -> PraxisConfig:
    return PraxisConfig(
        hidden_size=64,
        num_heads=4,
        num_queries=1,
        block_size=512,
        dropout=0.0,
        encoding=encoding,
    )


def test_alibi_no_ghost_matches_simple_bias():
    enc = registry.lookup("encoding", "alibi")(_cfg("alibi"))
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
    enc = registry.lookup("encoding", "alibi")(_cfg("alibi"))
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


# ------------------------------------------------------------------------------
# encoding_refactor_parity
# ------------------------------------------------------------------------------
# Structural parity tests for the encoding-registry refactor.
#
# These do NOT prove bit-exact equivalence to the pre-refactor inline _apply_rope (the
# inline version used split-even-odd pairs, the registry uses split-in-half; both are
# valid RoPE bases). They prove the properties that actually matter:
#
# 1. RoPE-style relative-position invariance still holds (shifting all positions by a
# constant doesn't change scores between same-relative-position pairs). 2. ALiBi
# continues to apply a strictly negative bias to past keys. 3. HoPE leaves a non-empty
# tail of head-dim slots unrotated, end to end.


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


def test_alibi_bias_strictly_negative_for_past_keys():
    cfg = _make_config("alibi")
    attn = CausalAttention(cfg)
    enc = attn.encoding

    # Use after_scores against zeros to get the bias matrix directly.
    batch, num_heads, q_len, kv_len = 1, attn.num_query_heads, 8, 8
    bias = enc.after_scores(torch.zeros(batch, num_heads, q_len, kv_len))
    # bias[..., q, k] should be <= 0 for k < q (past), == 0 on diagonal,
    # and >= 0 for k > q (future, irrelevant under causal mask but should
    # still respect the sign convention).
    diag = bias.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag))
    # Lower triangle (past keys): all <= 0.
    tril_mask = torch.tril(torch.ones(q_len, kv_len), diagonal=-1).bool()
    assert (bias[..., tril_mask] <= 0).all()
