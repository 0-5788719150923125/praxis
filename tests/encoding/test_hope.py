"""HoPE: the band-truncation cutoff matches the paper, the unrotated tail passes
through untouched, and HoPE collapses to RoPE when L is large enough that no band
crosses the threshold."""

import pytest
import torch

from praxis import PraxisConfig
from praxis.attention.causal import CausalAttention
from praxis.attention.infini import InfiniAttention
from praxis.encoding.hope import HoPE
from praxis.encoding.rope import RoPE

HEAD_DIM = 64
CPU = torch.device("cpu")


def _config(context_length):
    config = PraxisConfig(
        hidden_size=64,
        num_heads=4,
        num_queries=1,
        block_size=context_length,
        dropout=0.0,
        encoding="hope",
    )
    config.causal = True
    return config


@pytest.mark.parametrize(
    "context_length, pos_dim",
    [
        # threshold = 2pi/512 ~= 0.01227; inv_freq[i] = 10000^(-i/32) passes while
        # i < ~15.29, so bands 0..15 rotate (16 bands) => pos_dim 32.
        (512, 32),
        # threshold = 2pi/128 ~= 0.0491; the last band above it is index 10
        # (10000^(-10/32) ~= 0.0562), so 11 bands rotate => pos_dim 22.
        (128, 22),
        # The threshold is tiny enough that every band survives: plain RoPE.
        (2**30, HEAD_DIM),
    ],
)
def test_pos_dim_matches_the_threshold(context_length, pos_dim):
    enc = HoPE(_config(context_length))
    enc.log_theta_base.data.zero_()  # pin to theta=10000 for the documented math
    enc._compute_rope_embeddings(HEAD_DIM, 32, CPU, torch.float32)
    assert enc._pos_dim == pos_dim


def test_unrotated_tail_passes_through():
    enc = HoPE(_config(512))
    seq_len = 16
    enc._compute_rope_embeddings(HEAD_DIM, seq_len, CPU, torch.float32)
    x = torch.randn(1, 4, seq_len, HEAD_DIM)
    y = enc._apply_rotary_pos_emb(x, enc._cached_cos, enc._cached_sin)
    pos_dim = enc._pos_dim
    assert pos_dim < HEAD_DIM, "test assumes truncation; pick a smaller L if it fires"
    torch.testing.assert_close(y[..., pos_dim:], x[..., pos_dim:])
    assert not torch.allclose(y[..., :pos_dim], x[..., :pos_dim])


def test_matches_rope_when_threshold_keeps_all_bands():
    seq_len = 16
    config = _config(2**30)
    hope, rope = HoPE(config), RoPE(config)
    hope._compute_rope_embeddings(HEAD_DIM, seq_len, CPU, torch.float32)
    rope._compute_rope_embeddings(HEAD_DIM, seq_len, CPU, torch.float32)
    torch.manual_seed(0)
    x = torch.randn(2, 4, seq_len, HEAD_DIM)
    y_hope = hope._apply_rotary_pos_emb(x, hope._cached_cos, hope._cached_sin)
    y_rope = rope._apply_rotary_pos_emb(x, rope._cached_cos, rope._cached_sin)
    torch.testing.assert_close(y_hope, y_rope)


@pytest.mark.parametrize("host", [CausalAttention, InfiniAttention])
def test_cutoff_resolves_through_an_attention_forward(host):
    """The band cutoff is resolved on first call, so a forward through each host
    proves the registry path really reached ``before_scores``."""
    attn = host(_config(256)).eval()
    assert attn.encoding._pos_dim is None
    with torch.no_grad():
        attn(inputs=torch.randn(1, 32, 64))
    assert 0 < attn.encoding._pos_dim < attn.head_dim
