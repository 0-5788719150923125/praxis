"""HelicalSparseWidth: the true-slice variant, which narrows the matmuls (and any
per-channel activation parameters) instead of masking, and drops attention heads."""

import torch
import torch.nn as nn

from praxis import PraxisConfig, registry
from praxis.attention.arc import ArcAttention


def test_sparse_slices_the_matmul_and_grads(glu):
    """At a deflated step the up matmul emits 2r rows and down consumes r cols,
    and only those receive gradient (the rest of the weight is untouched)."""
    pol = registry.lookup("width", "helical_sparse")()
    blk, x = glu(), torch.randn(2, 4, 16)
    blk.zero_grad()
    with pol.scope([blk], current_depth=5, max_depth=6):  # frac 0.25 -> r=6
        out = blk(x)
        out.pow(2).mean().backward()
    r = 6
    assert (blk.up.weight.grad.abs().sum(1) > 0).sum().item() == 2 * r
    assert (blk.down.weight.grad.abs().sum(0) > 0).sum().item() == r


class _ParamAct(nn.Module):
    """Per-channel parametric activation, like Serpent's a/b/g."""

    def __init__(self, width):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * self.scale  # broadcasts on last dim; breaks if x desyncs


def test_sparse_slices_parametric_activation_in_sync(glu):
    """A GLU with a per-channel parametric activation IS sliced - its activation
    param is sliced to the same window, so the forward stays well formed and the
    param's gradient lands only on the active channels."""
    pol = registry.lookup("width", "helical_sparse")()
    blk, x = glu(act=_ParamAct(24)), torch.randn(2, 4, 16)
    blk.zero_grad()
    with pol.scope([blk], current_depth=5, max_depth=6):  # frac 0.25 -> r=6
        out = blk(x)  # would raise if act param desynced from the sliced inner dim
        out.pow(2).mean().backward()
    assert out.shape == (2, 4, 16)
    assert (blk.act.scale.grad.abs() > 0).sum().item() == 6  # only the active slice
    assert blk.act.scale.shape == (24,)  # restored full width


def test_sparse_defers_while_activation_is_lazy():
    """A still-lazy per-channel activation param disables slicing for that GLU,
    so the param materializes at full width before it is ever sliced."""
    from torch.nn.parameter import UninitializedParameter

    from praxis.width.sparse import _activation_channel_tensors

    act = _ParamAct(24)
    assert _activation_channel_tensors(act, 24) == [(act._parameters, "scale")]
    act.scale = UninitializedParameter()
    assert _activation_channel_tensors(act, 24) is None  # bail while lazy


def test_sparse_policy_drops_heads_in_a_block():
    """The sparse policy reaches attention through a containing module."""
    pol = registry.lookup("width", "helical_sparse")()
    attn = ArcAttention(
        PraxisConfig(hidden_size=128, num_heads=4, num_queries=2, depth=8, dropout=0.0)
    )
    holder = nn.Module()
    holder.attn = attn
    x = torch.randn(2, 12, 128)
    with pol.scope([holder], current_depth=5, max_depth=6):  # frac 0.25 -> 1 head
        assert attn.num_heads == 1
        out, *_ = attn(x, current_depth=5)
        assert out.shape == (2, 12, 128)
    assert attn.num_heads == 4
