"""ALiBi: the FlexAttention score_mod closure and the materialized bias agree on a
linear penalty for past keys, with the ghost column left unbiased."""

import pytest
import torch

from praxis import PraxisConfig
from praxis.encoding.alibi import ALiBi


def _alibi():
    config = PraxisConfig(
        hidden_size=64, num_heads=4, num_queries=1, block_size=512, dropout=0.0
    )
    return ALiBi(config)


@pytest.mark.parametrize("ghost_offset", [0, 1])
def test_score_mod_adds_the_slope_times_the_distance(ghost_offset):
    """With a ghost column prepended (offset 1), key index 0 is the ghost and gets
    no bias, and every real key sits one index later."""
    enc = _alibi()
    device = torch.device("cpu")
    mod = enc.build_score_mod(num_heads=4, device=device, ghost_offset=ghost_offset)
    slopes = enc.compute_slopes(4, device)
    for h, q_idx, kv_idx in [(0, 0, 0), (1, 5, 3), (2, 4, 1), (3, 7, 9)]:
        score = torch.tensor(0.7)
        if ghost_offset and kv_idx == 0:
            expected = score
        else:
            expected = score + slopes[h] * (kv_idx - ghost_offset - q_idx)
        got = mod(
            score, b=0, h=h, q_idx=torch.tensor(q_idx), kv_idx=torch.tensor(kv_idx)
        )
        torch.testing.assert_close(got, expected)


def test_bias_is_zero_on_the_diagonal_and_negative_for_past_keys():
    enc = _alibi()
    q_len = kv_len = 8
    bias = enc.after_scores(torch.zeros(1, 4, q_len, kv_len))
    diag = bias.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag))
    past = torch.tril(torch.ones(q_len, kv_len), diagonal=-1).bool()
    assert (bias[..., past] < 0).all()
