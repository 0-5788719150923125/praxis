"""Unit tests for HoPE encoding.

Confirms the band-truncation cutoff matches the paper, the unrotated tail
passes through untouched, and HoPE collapses to RoPE when L is large enough
that no bands cross the threshold.
"""

import math

import pytest
import torch

from praxis import PraxisConfig
from praxis.encoding import ENCODING_REGISTRY
from praxis.encoding.hope import HoPE
from praxis.encoding.rope import RoPE


def _make_config(context_length: int, num_heads: int = 4, num_queries: int = 1):
    return PraxisConfig(
        hidden_size=64,
        num_heads=num_heads,
        num_queries=num_queries,
        block_size=context_length,
        dropout=0.0,
        encoding="hope",
    )


def test_registered_in_registry():
    assert "hope" in ENCODING_REGISTRY
    assert ENCODING_REGISTRY["hope"] is HoPE


def test_pos_dim_matches_threshold_512_head64():
    # head_dim=64, theta=10000, L=512 -> threshold = 2pi/512 ~= 0.01227
    # inv_freq[i] = 10000^(-i/32). Threshold satisfied while i/32 < log10(0.01227)/-4,
    # i.e. i < ~15.29. Bands 0..15 pass (16 bands), 16..31 drop => pos_dim = 32.
    cfg = _make_config(context_length=512)
    enc = HoPE(cfg)
    enc.log_theta_base.data.zero_()  # pin to theta=10000 for the documented math
    head_dim = 64
    enc._compute_rope_embeddings(
        head_dim=head_dim,
        seq_len=32,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert enc._pos_dim == 32, f"expected 32 rotated dims at L=512, got {enc._pos_dim}"


def test_pos_dim_drops_more_at_smaller_L():
    # L=128 is more aggressive: threshold = 2pi/128 ~= 0.0491. The last band
    # whose inv_freq exceeds that is index 10 (10000^(-10/32) ~= 0.0562 > 0.0491),
    # so 11 bands pass => pos_dim = 22, 42 dims unrotated.
    cfg = _make_config(context_length=128)
    enc = HoPE(cfg)
    enc.log_theta_base.data.zero_()  # pin to theta=10000 for the documented math
    enc._compute_rope_embeddings(
        head_dim=64,
        seq_len=32,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert enc._pos_dim == 22, f"expected 22 rotated dims at L=128, got {enc._pos_dim}"


def test_pos_dim_collapses_to_rope_at_large_L():
    # At L=2**30, the threshold is tiny enough that every band survives,
    # so HoPE should behave exactly like RoPE.
    cfg = _make_config(context_length=2**30)
    enc = HoPE(cfg)
    head_dim = 64
    enc._compute_rope_embeddings(
        head_dim=head_dim,
        seq_len=32,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert enc._pos_dim == head_dim


def test_unrotated_tail_passes_through():
    # The slots past pos_dim must come out unchanged; only the prefix gets rotated.
    cfg = _make_config(context_length=512)
    enc = HoPE(cfg)
    head_dim = 64
    seq_len = 16
    enc._compute_rope_embeddings(
        head_dim=head_dim,
        seq_len=seq_len,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    x = torch.randn(1, 4, seq_len, head_dim)
    y = enc._apply_rotary_pos_emb(x, enc._cached_cos, enc._cached_sin)
    pos_dim = enc._pos_dim
    assert pos_dim < head_dim, "test assumes truncation; pick a smaller L if it fires"
    torch.testing.assert_close(y[..., pos_dim:], x[..., pos_dim:])
    # And the rotated prefix should differ at seq positions > 0.
    assert not torch.allclose(y[..., :pos_dim], x[..., :pos_dim])


def test_matches_rope_when_threshold_keeps_all_bands():
    # When pos_dim == head_dim, HoPE should match RoPE bit-for-bit
    # (modulo the all-true mask path being a no-op).
    head_dim = 64
    seq_len = 16
    cfg = _make_config(context_length=2**30)
    hope = HoPE(cfg)
    rope = RoPE(cfg)
    hope._compute_rope_embeddings(head_dim, seq_len, torch.device("cpu"), torch.float32)
    rope._compute_rope_embeddings(head_dim, seq_len, torch.device("cpu"), torch.float32)
    torch.manual_seed(0)
    x = torch.randn(2, 4, seq_len, head_dim)
    y_hope = hope._apply_rotary_pos_emb(x, hope._cached_cos, hope._cached_sin)
    y_rope = rope._apply_rotary_pos_emb(x, rope._cached_cos, rope._cached_sin)
    torch.testing.assert_close(y_hope, y_rope)


def test_before_scores_runs_through_registry_interface():
    # End-to-end smoke: HoPE plugs into the standard before_scores signature
    # used by syntaxes.py and modular.py.
    cfg = _make_config(context_length=512, num_heads=4, num_queries=1)
    enc = ENCODING_REGISTRY["hope"](cfg)
    batch, num_heads, seq_len, head_dim = 2, 4, 32, 64
    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)
    v = torch.randn(batch, num_heads, seq_len, head_dim)
    q2, k2, v2 = enc.before_scores(q, k, v)
    assert q2.shape == q.shape
    assert k2.shape == k.shape
    assert v2 is v  # values pass through untouched
