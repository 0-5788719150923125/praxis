"""RoPE: scores depend only on relative position."""

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
