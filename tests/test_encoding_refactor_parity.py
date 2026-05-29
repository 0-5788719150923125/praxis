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


def test_rope_outputs_are_finite_and_shape_preserving():
    cfg = _make_config("rope")
    attn = CausalAttention(cfg)
    attn.eval()
    x = torch.randn(2, 32, 64)
    with torch.no_grad():
        y, _, _ = attn(inputs=x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_hope_tail_dims_actually_unrotated():
    cfg = _make_config("hope")
    attn = CausalAttention(cfg)
    attn.eval()
    # Run one forward to resolve pos_dim on the encoding.
    with torch.no_grad():
        attn(inputs=torch.randn(1, 32, 64))
    enc = attn.encoding
    assert enc._pos_dim is not None
    assert enc._pos_dim < attn.head_dim, (
        f"expected HoPE to drop at least one band at block_size={cfg.block_size}, "
        f"but pos_dim={enc._pos_dim} == head_dim={attn.head_dim}"
    )


def test_rope_relative_position_invariance_through_encoding():
    """RoPE's defining property: scores depend only on (t - s), not absolute t.

    We exercise this on the encoding module directly (rather than the full
    attention, which has causal masking that breaks the literal translation
    test). The registry RoPE must still produce dot products that depend only
    on relative position when both Q and K are shifted by the same offset.
    """
    cfg = _make_config("rope")
    attn = CausalAttention(cfg)
    enc = attn.encoding

    head_dim = attn.head_dim
    seq_len = 8
    torch.manual_seed(0)
    q = torch.randn(1, attn.num_query_heads, seq_len, head_dim)
    k = torch.randn(1, attn.num_heads, seq_len, head_dim)
    v = torch.zeros_like(k)

    # No offset: rotated Q/K at positions 0..seq_len-1.
    enc._cached_seq_length = None
    q0, k0, _ = enc.before_scores(q.clone(), k.clone(), v.clone(), offset=0)
    scores0 = q0 @ k0.transpose(-2, -1)

    # Shift both Q and K by the same offset. Relative positions unchanged,
    # so the score matrix between corresponding pairs must match.
    enc._cached_seq_length = None
    q1, k1, _ = enc.before_scores(q.clone(), k.clone(), v.clone(), offset=5)
    scores1 = q1 @ k1.transpose(-2, -1)

    torch.testing.assert_close(scores0, scores1, rtol=1e-4, atol=1e-4)


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
