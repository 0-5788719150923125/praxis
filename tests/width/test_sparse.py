"""Mixture-of-widths: the helical deflation policy and its profile."""

import torch
import torch.nn as nn

from praxis import registry

# ─── Sparse (true-slice) variant ─────────────────────────────────────────────


class _GLU(nn.Module):
    """up -> chunk(2) -> a*act(b) -> down, like GatedLinearMLP/ArcGLU."""

    def __init__(self, hidden=16, inner=24):
        super().__init__()
        self.up = nn.Linear(hidden, 2 * inner)
        self.down = nn.Linear(inner, hidden)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(a * torch.tanh(b))


def test_sparse_slices_the_matmul_and_grads():
    """At a deflated step the up matmul emits 2r rows and down consumes r cols,
    and only those receive gradient (the rest of the weight is untouched)."""
    pol = registry.lookup("width", "helical_sparse")()
    blk, x = _GLU(hidden=16, inner=24), torch.randn(2, 4, 16)
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


class _ParamGLU(nn.Module):
    def __init__(self, hidden=16, inner=24):
        super().__init__()
        self.up = nn.Linear(hidden, 2 * inner)
        self.act = _ParamAct(inner)
        self.down = nn.Linear(inner, hidden)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(a * self.act(b))


def test_sparse_slices_parametric_activation_in_sync():
    """A GLU with a per-channel parametric activation IS sliced - its activation
    param is sliced to the same window, so the forward stays well formed and the
    param's gradient lands only on the active channels."""
    pol = registry.lookup("width", "helical_sparse")()
    blk, x = _ParamGLU(hidden=16, inner=24), torch.randn(2, 4, 16)
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


# ─── Attention head-drop recipe ──────────────────────────────────────────────


def _arc_attention(hidden=128, num_heads=4, num_queries=2):
    from praxis import PraxisConfig
    from praxis.attention.arc import ArcAttention

    cfg = PraxisConfig(
        hidden_size=hidden,
        num_heads=num_heads,
        num_queries=num_queries,
        depth=8,
        dropout=0.0,
        encoding="rope",
        causal=False,
    )
    return ArcAttention(cfg)


def test_sparse_policy_drops_heads_in_a_block():
    """The sparse policy reaches attention through a containing module."""
    pol = registry.lookup("width", "helical_sparse")()
    attn = _arc_attention()
    holder = nn.Module()
    holder.attn = attn
    x = torch.randn(2, 12, 128)
    with pol.scope([holder], current_depth=5, max_depth=6):  # frac 0.25 -> 1 head
        assert attn.num_heads == 1
        out, *_ = attn(x, current_depth=5)
        assert out.shape == (2, 12, 128)
    assert attn.num_heads == 4
