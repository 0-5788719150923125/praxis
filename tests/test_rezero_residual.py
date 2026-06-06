"""ReZero residual: per-depth zero-init gains on the block branches."""

import torch

from praxis import PraxisConfig
from praxis.blocks.transformer import TransformerBlock
from praxis.residuals import RESIDUAL_REGISTRY, ReZeroConnection


def _cfg(**over):
    base = dict(
        vocab_size=256,
        hidden_size=64,
        embed_size=64,
        num_heads=4,
        num_queries=1,
        depth=3,
        num_layers=3,
        residual_type="rezero",
    )
    base.update(over)
    return PraxisConfig(**base)


def test_registered():
    assert RESIDUAL_REGISTRY["rezero"] is ReZeroConnection


def test_identity_at_init_per_depth():
    res = ReZeroConnection(8, num_depths=3)
    h, branch = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    for d in range(4):  # depths past num_depths clamp to the last gain
        torch.testing.assert_close(
            res.connect_depth(h, branch, None, current_depth=d), h
        )


def test_per_depth_gains_are_independent():
    res = ReZeroConnection(8, num_depths=3)
    with torch.no_grad():
        res.alpha[1] = 0.5
    h, branch = torch.zeros(1, 4, 8), torch.ones(1, 4, 8)
    assert res.connect_depth(h, branch, None, current_depth=0).abs().sum() == 0
    torch.testing.assert_close(
        res.connect_depth(h, branch, None, current_depth=1), 0.5 * branch
    )


def test_gain_gets_gradient_only_at_its_depth():
    res = ReZeroConnection(8, num_depths=3)
    h, branch = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    res.connect_depth(h, branch, None, current_depth=2).sum().backward()
    assert res.alpha.grad[2] != 0
    assert res.alpha.grad[0] == 0 and res.alpha.grad[1] == 0


def test_transformer_block_is_identity_at_init():
    torch.manual_seed(0)
    block = TransformerBlock(_cfg())
    x = torch.randn(2, 8, 64)
    out, _, _, _ = block(x, None, current_depth=1)
    torch.testing.assert_close(out, x)


def test_transformer_block_learns_force():
    torch.manual_seed(0)
    block = TransformerBlock(_cfg())
    with torch.no_grad():
        block.attn_res.alpha[1] = 0.3
        block.ffn_res.alpha[1] = 0.3
    x = torch.randn(2, 8, 64)
    out, _, _, _ = block(x, None, current_depth=1)
    assert not torch.allclose(out, x)
    # Other depths are still fully dampened.
    out0, _, _, _ = block(x, None, current_depth=0)
    torch.testing.assert_close(out0, x)
