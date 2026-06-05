"""Mixture-of-widths: the helical deflation policy and its profile."""

import torch
import torch.nn as nn

from praxis.width import WIDTH_REGISTRY
from praxis.width.helical import width_fraction


class _Block(nn.Module):
    """A GLU-shaped stand-in: ``down`` is the inner projection width policies mask."""

    def __init__(self, hidden=8, inner=12):
        super().__init__()
        self.up = nn.Linear(hidden, 2 * inner)
        self.down = nn.Linear(inner, hidden)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(a * b)


def _active_channels(block, policy, depth, max_depth, x):
    """Count inner channels that survive the policy's mask at this depth."""
    seen = {}
    with policy.scope([block], current_depth=depth, max_depth=max_depth):
        # Register AFTER the policy hook so we observe the masked input.
        handle = block.down.register_forward_pre_hook(
            lambda m, a: seen.setdefault("x", a[0].detach().clone())
        )
        block(x)
        handle.remove()
    return int((seen["x"].abs().sum(dim=(0, 1)) > 0).sum().item())


def test_registry_keys():
    assert "none" in WIDTH_REGISTRY and "helical" in WIDTH_REGISTRY


def test_full_width_is_noop_and_has_no_profile():
    policy = WIDTH_REGISTRY["none"]()
    assert policy.profile(8) is None
    block, x = _Block(), torch.randn(2, 3, 8)
    assert _active_channels(block, policy, 0, 8, x) == 12  # nothing masked


def test_profile_is_an_arch():
    """Inflate early, decay through the tail: ends sit at the floor, the crest
    near the front rises well above them."""
    prof = WIDTH_REGISTRY["helical"]().profile(6)
    assert abs(prof[0] - 0.25) < 1e-6 and abs(prof[-1] - 0.25) < 1e-6
    assert max(prof) > 0.9
    assert prof.index(max(prof)) < len(prof) // 2  # crest is in the front half


def test_deflation_matches_profile():
    policy = WIDTH_REGISTRY["helical"]()
    block, x = _Block(), torch.randn(2, 3, 8)
    prof = policy.profile(6)
    for d in range(6):
        expected = max(1, min(12, round(prof[d] * 12)))
        assert _active_channels(block, policy, d, 6, x) == expected


def test_hooks_are_removed_on_exit():
    policy = WIDTH_REGISTRY["helical"]()
    block, x = _Block(), torch.randn(2, 3, 8)
    with policy.scope([block], current_depth=3, max_depth=6):
        pass
    assert _active_channels(block, WIDTH_REGISTRY["none"](), 0, 6, x) == 12


def test_helix_window_precesses_with_depth():
    """The active set at successive depths is rotated, not identical (coverage)."""
    policy = WIDTH_REGISTRY["helical_steady"]()  # constant width, so only the start moves
    block, x = _Block(), torch.randn(2, 3, 8)

    def active_set(depth):
        seen = {}
        with policy.scope([block], current_depth=depth, max_depth=8):
            handle = block.down.register_forward_pre_hook(
                lambda m, a: seen.setdefault("x", a[0].detach().clone())
            )
            block(x)
            handle.remove()
        return set((seen["x"].abs().sum(dim=(0, 1)) > 0).nonzero().flatten().tolist())

    assert active_set(0) != active_set(1)


def test_width_fraction_single_depth():
    assert width_fraction(0, 1, 0.25, 0.3) == 1.0  # degenerate stack = full width


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


def test_sparse_preserves_output_shape():
    pol = WIDTH_REGISTRY["helical_sparse"]()
    blk, x = _GLU(), torch.randn(2, 4, 16)
    for d in range(6):
        with pol.scope([blk], current_depth=d, max_depth=6):
            out = blk(x)
        assert out.shape == (2, 4, 16)


def test_sparse_slices_the_matmul_and_grads():
    """At a deflated step the up matmul emits 2r rows and down consumes r cols,
    and only those receive gradient (the rest of the weight is untouched)."""
    pol = WIDTH_REGISTRY["helical_sparse"]()
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
    pol = WIDTH_REGISTRY["helical_sparse"]()
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


def test_sparse_restores_full_forward_on_exit():
    pol = WIDTH_REGISTRY["helical_sparse"]()
    blk, x = _GLU(inner=24), torch.randn(2, 4, 16)
    with pol.scope([blk], current_depth=5, max_depth=6):
        pass
    assert blk.up(x).shape[-1] == 48  # back to full 2*inner


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


def test_head_budget_preserves_output_and_restores():
    attn = _arc_attention()
    x = torch.randn(2, 12, 128)
    with attn.head_budget(torch.tensor([1])):
        assert attn.num_heads == 1 and attn.num_query_heads == 2
        out, *_ = attn(x, current_depth=3)
        assert out.shape == (2, 12, 128)  # residual stream stays full width
    assert attn.num_heads == 4 and attn.num_query_heads == 8  # restored


def test_head_budget_grads_only_kept_heads():
    """Keeping 1 of 4 KV heads should grad exactly its channels: 2 query heads
    (GQA) + 1 K + 1 V, times head_dim, of the fused QKV; and only the kept query
    heads of the per-head betas."""
    attn = _arc_attention()
    x = torch.randn(2, 12, 128)
    attn.zero_grad()
    with attn.head_budget(torch.tensor([2])):
        attn(x, current_depth=5)[0].pow(2).mean().backward()
    hd = attn.head_dim
    assert (attn.qkv.weight.grad.abs().sum(1) > 0).sum().item() == (2 + 1 + 1) * hd
    assert (attn.betas.grad.abs().sum((0, 2, 3)) > 0).sum().item() == 2


def test_sparse_policy_drops_heads_in_a_block():
    """The sparse policy reaches attention through a containing module."""
    pol = WIDTH_REGISTRY["helical_sparse"]()
    attn = _arc_attention()
    holder = nn.Module()
    holder.attn = attn
    x = torch.randn(2, 12, 128)
    with pol.scope([holder], current_depth=5, max_depth=6):  # frac 0.25 -> 1 head
        assert attn.num_heads == 1
        out, *_ = attn(x, current_depth=5)
        assert out.shape == (2, 12, 128)
    assert attn.num_heads == 4
