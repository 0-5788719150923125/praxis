"""RoPE: scores depend only on relative position."""

import pytest
import torch

from praxis import PraxisConfig
from praxis.encoding.rope import RoPE


def test_scores_are_invariant_to_a_shared_offset():
    """Shifting Q and K by the same offset leaves every q.k score unchanged."""
    enc = RoPE(PraxisConfig(hidden_size=64, num_heads=4, num_queries=1, block_size=256))
    torch.manual_seed(0)
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 4, 8, 16)
    v = torch.zeros_like(k)
    q0, k0, _ = enc.before_scores(q.clone(), k.clone(), v, offset=0)
    q1, k1, _ = enc.before_scores(q.clone(), k.clone(), v, offset=5)
    torch.testing.assert_close(
        q0 @ k0.transpose(-2, -1), q1 @ k1.transpose(-2, -1), rtol=1e-4, atol=1e-4
    )
    # ...while the rotation itself did move with the offset.
    assert not torch.allclose(q0, q1)


def test_theta_starts_at_the_standard_10000():
    config = PraxisConfig(hidden_size=64, num_heads=4, num_queries=1, block_size=256)
    assert RoPE(config).theta == pytest.approx(10_000.0, rel=1e-4)


def _pair_angles(enc, head_dim, positions=6):
    """Rotation angle each rotate-half pair receives at each position."""
    x = torch.zeros(1, 1, positions, head_dim)
    x[..., : head_dim // 2] = 1.0
    q, _, _ = enc.before_scores(x, x, torch.zeros_like(x))
    half = head_dim // 2
    return torch.atan2(q[0, 0, :, (head_dim + 1) // 2 :], q[0, 0, :, :half])


def test_matches_the_transformers_llama_reference():
    """Same inv_freq, same rotate-half convention: transformers' Llama
    ``apply_rotary_pos_emb`` (cos/sin as ``cat(freqs, freqs)``) is the reference."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    config = PraxisConfig(hidden_size=64, num_heads=4, num_queries=1, block_size=256)
    enc = RoPE(config)
    torch.manual_seed(0)
    q = torch.randn(2, 4, 10, 16)
    k = torch.randn(2, 4, 10, 16)
    ours_q, ours_k, _ = enc.before_scores(q, k, torch.zeros_like(k))

    inv_freq = enc._compute_inv_freq(16, q.device, 0).detach()
    freqs = torch.arange(10).float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)[None]
    ref_q, ref_k = apply_rotary_pos_emb(q, k, emb.cos(), emb.sin())
    torch.testing.assert_close(ours_q, ref_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ours_k, ref_k, rtol=1e-5, atol=1e-5)


def test_every_band_turns_its_own_pair():
    """Pair j turns at band j: all head_dim/2 rates, each once, all positive."""
    config = PraxisConfig(hidden_size=64, num_heads=4, num_queries=1, block_size=256)
    enc = RoPE(config)
    angles = _pair_angles(enc, 8)
    inv_freq = enc._compute_inv_freq(8, torch.device("cpu"), 0).detach()
    expected = torch.arange(6).float()[:, None] * inv_freq[None, :]
    torch.testing.assert_close(angles, torch.atan2(expected.sin(), expected.cos()))


def test_odd_head_dim_leaves_the_unpaired_slot_unrotated():
    config = PraxisConfig(hidden_size=63, num_heads=3, num_queries=1, block_size=256)
    enc = RoPE(config)
    torch.manual_seed(0)
    q = torch.randn(1, 3, 7, 21)
    out, _, _ = enc.before_scores(q, q.clone(), torch.zeros_like(q))
    torch.testing.assert_close(out[..., 10], q[..., 10])  # x1's last slot
    q0, k0, _ = enc.before_scores(q.clone(), q.clone(), torch.zeros_like(q), offset=0)
    q1, k1, _ = enc.before_scores(q.clone(), q.clone(), torch.zeros_like(q), offset=4)
    torch.testing.assert_close(
        q0 @ k0.transpose(-2, -1), q1 @ k1.transpose(-2, -1), rtol=1e-4, atol=1e-4
    )
