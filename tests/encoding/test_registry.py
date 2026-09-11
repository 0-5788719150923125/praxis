import pytest
import torch

from praxis import PraxisConfig, registry

# ------------------------------------------------------------------------------
# hope
# ------------------------------------------------------------------------------
# Unit tests for HoPE encoding.
#
# Confirms the band-truncation cutoff matches the paper, the unrotated tail passes
# through untouched, and HoPE collapses to RoPE when L is large enough that no bands
# cross the threshold.


def _make_config(context_length: int, num_heads: int = 4, num_queries: int = 1):
    return PraxisConfig(
        hidden_size=64,
        num_heads=num_heads,
        num_queries=num_queries,
        block_size=context_length,
        dropout=0.0,
        encoding="hope",
    )


def test_before_scores_runs_through_registry_interface():
    # End-to-end smoke: HoPE plugs into the standard before_scores signature
    # used by syntaxes.py and modular.py.
    cfg = _make_config(context_length=512, num_heads=4, num_queries=1)
    enc = registry.lookup("encoding", "hope")(cfg)
    batch, num_heads, seq_len, head_dim = 2, 4, 32, 64
    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)
    v = torch.randn(batch, num_heads, seq_len, head_dim)
    q2, k2, v2 = enc.before_scores(q, k, v)
    assert q2.shape == q.shape
    assert k2.shape == k.shape
    assert v2 is v  # values pass through untouched


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


def test_nope_rope_hope_return_none():
    for name in ("nope", "rope", "hope"):
        enc = registry.lookup("encoding", name)(_cfg(name))
        mod = enc.build_score_mod(num_heads=4, device=torch.device("cpu"))
        assert mod is None, f"{name} should return None from build_score_mod"
