"""Ghost features: the algebra, the in-place parametrization, and the walker.

The properties pinned here are the ones the experiment's interpretation rests on.
If the complex product is wrong the arm is not testing the paper's mechanism; if
the init does not inherit the host module's scale the arms differ by an init as
well as a mechanism; if the walker misses a target the run is not the experiment.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from praxis.transforms.algebra import ALGEBRAS
from praxis.transforms.ghost import (
    AUTO_ORDER,
    GhostExpansion,
    ghost_parameter,
    pick_algebra,
)

CONV_SHAPE = (544, 272, 3)  # abstractinator-o's ConvBlock.conv
PEER_SHAPE = (729, 272)  # PEER's default bank: gcd(729, 272) == 1


def test_complex_expansion_reproduces_the_complex_product():
    """Re{x.w} = x0 w0 - x1 w1 and Im{x.w} = x0 w1 + x1 w0, exactly."""
    torch.manual_seed(0)
    p = GhostExpansion("complex", (4, 4))
    real = torch.randn(2, 4)
    full = p(real)
    for row in range(2):
        w0, w1 = real[row, 0], real[row, 1]
        assert torch.allclose(full[row, :2], torch.stack([w0, -w1]), atol=1e-6)
        assert torch.allclose(full[2 + row, :2], torch.stack([w1, w0]), atol=1e-6)


def test_quaternion_expansion_matches_the_multiplication_table():
    torch.manual_seed(0)
    p = GhostExpansion("quaternion", (8, 4))
    real = torch.randn(2, 4)
    full = p(real)
    for r in range(2):
        w0, w1, w2, w3 = real[r]
        expected = [
            (w0, -w1, -w2, -w3),
            (w1, w0, w3, -w2),
            (w2, -w3, w0, w1),
            (w3, w2, -w1, w0),
        ]
        for k, comps in enumerate(expected):
            assert torch.allclose(full[k * 2 + r], torch.stack(comps), atol=1e-6)


# --- the parametrization ----------------------------------------------------


@pytest.mark.parametrize(
    "algebra,ratio", [("complex", 0.5), ("quaternion", 0.25), ("cyclic3", 1 / 3)]
)
def test_budget_is_exactly_one_over_d(algebra, ratio):
    d = len(ALGEBRAS[algebra][0])
    shape = (4 * d, 4 * d, 3)
    p = GhostExpansion(algebra, shape)
    assert p.real_numel == pytest.approx(math.prod(shape) * ratio, rel=1e-9)
    assert tuple(p(torch.randn(shape[0] // d, shape[1], shape[2])).shape) == shape


def test_expansion_is_full_rank_and_that_is_the_mechanism():
    """`P_k` acts on the INPUT axis, so the d expanded blocks are NOT linear
    combinations of the out/d real rows - each applies a different input-space
    transform, and the stack comes out FULL rank. Any output-axis factorization
    at the same budget is capped at params/(out+fan). That asymmetry is why the
    honest control is another signed permutation, not another compression."""
    torch.manual_seed(0)
    p = GhostExpansion("complex", CONV_SHAPE)
    real = torch.randn(CONV_SHAPE[0] // 2, CONV_SHAPE[1], CONV_SHAPE[2])
    W = p(real).reshape(CONV_SHAPE[0], -1)
    assert torch.linalg.matrix_rank(W).item() == CONV_SHAPE[0]
    matched_rank = p.real_numel // (CONV_SHAPE[0] + CONV_SHAPE[1] * CONV_SHAPE[2])
    assert matched_rank < CONV_SHAPE[0]


def test_right_inverse_is_the_least_squares_fit():
    """Each P_k used here is an involution, so mean_k P_k(W_k) is the LS real
    tensor. This is what makes a ghosted module inherit its HOST's init instead
    of a scale chosen here - the bug that voided the first -r run was exactly a
    hand-reproduced init, at 0.045x."""
    torch.manual_seed(0)
    p = GhostExpansion("complex", (8, 6))
    real = torch.randn(4, 6)
    # A weight already ON the manifold must round-trip exactly.
    assert torch.allclose(p.right_inverse(p(real)), real, atol=1e-6)


@pytest.mark.parametrize("shape", [CONV_SHAPE, (256, 128, 3), (272, 272), (512, 272)])
def test_init_scale_inherits_the_host_module(shape):
    """The expanded weight must start near the scale the host module chose, for
    every host type, with no scale factor picked in the ghost code."""
    torch.manual_seed(0)
    out, in_, *tail = shape
    host = nn.Conv1d(in_, out, tail[0]) if tail else nn.Linear(in_, out)
    ref = host.weight.detach().std().item()
    ghost_parameter(host, "weight", "complex")
    got = host.weight.detach().std().item()
    assert 0.5 < got / ref < 1.5


def test_indivisible_shapes_are_refused_not_approximated():
    """PEER's default bank is the canonical case: [729, 272] with 729 = 3^6 and
    272 = 2^4 * 17, so gcd == 1 and NO d > 1 divides both axes. The fix belongs to
    PEER's key count, which it requests through `aligned_size` - never a padded or
    partial expansion here."""
    assert pick_algebra(PEER_SHAPE, "auto") is None
    assert pick_algebra(PEER_SHAPE, "complex") is None
    assert pick_algebra((676, 272), "auto") in AUTO_ORDER
    with pytest.raises(ValueError):
        GhostExpansion("complex", (7, 4))
    with pytest.raises(ValueError):
        GhostExpansion("complex", (4, 7))


def test_auto_takes_the_deepest_cut_that_divides():
    assert pick_algebra((544, 272), "auto") == "quaternion"  # d=4 divides both
    assert pick_algebra((90, 272), "auto") == "complex"  # 90 % 4 != 0


def test_random_control_meets_the_same_conditions_and_is_reproducible():
    """Otherwise the control tests degeneracy rather than arbitrariness. Seeded
    from the target's qualname via crc32, because hash() is salted per process
    and would reroll the control on restart."""
    a = GhostExpansion("complex", (8, 6), tag="site.a", randomize=True)
    b = GhostExpansion("complex", (8, 6), tag="site.a", randomize=True)
    c = GhostExpansion("complex", (8, 6), tag="site.b", randomize=True)
    assert torch.equal(a.structure, b.structure)
    assert not torch.equal(a.structure, c.structure)
    assert a.algebra.startswith("random")


# --- host module equivalence ------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: nn.Linear(272, 544),
        lambda: nn.Conv1d(272, 544, 3, dilation=2, padding=4),
        lambda: nn.Embedding(512, 272),
        lambda: nn.EmbeddingBag(512, 272, mode="sum"),
    ],
)
def test_host_forward_matches_the_same_module_given_the_expanded_weight(build):
    """One mechanism, every host type: the module's own forward reads `weight`,
    so a derived `weight` needs no per-type wrapper. EmbeddingBag included -
    the expansion materializes the full weight BEFORE the bag reduction, so the
    'sums rows before the flip can reach them' objection does not apply."""
    torch.manual_seed(0)
    ghosted, plain = build(), build()
    ghost_parameter(ghosted, "weight", "complex")
    with torch.no_grad():
        plain.weight.copy_(ghosted.weight)
        if getattr(plain, "bias", None) is not None:
            plain.bias.copy_(ghosted.bias)
    if isinstance(ghosted, (nn.Embedding, nn.EmbeddingBag)):
        x = torch.randint(0, 512, (2, 6))
    elif isinstance(ghosted, nn.Conv1d):
        x = torch.randn(2, 272, 20)
    else:
        x = torch.randn(2, 272)
    assert torch.allclose(ghosted(x), plain(x), atol=1e-5)


def test_ghosted_causal_conv_stays_causal():
    """ConvBlock trims `padding` after the conv; the wrapper must preserve
    stride/padding/dilation exactly or causality silently breaks."""
    torch.manual_seed(0)
    for dil, pad in ((1, 2), (2, 4), (4, 8)):
        conv = nn.Conv1d(16, 32, 3, dilation=dil, padding=pad)
        ghost_parameter(conv, "weight", "complex")
        x = torch.randn(2, 16, 40)
        y1 = conv(x)[..., :-pad]
        x2 = x.clone()
        x2[..., -1] += 10.0
        y2 = conv(x2)[..., :-pad]
        assert torch.allclose(y1[..., :-1], y2[..., :-1], atol=1e-6)


def test_module_identity_is_preserved_which_is_why_it_composes():
    """The reason the transform mutates instead of wrapping. SMEAR registers its
    MergedLinear both at the block qualname AND in its own `wrappers` dict, so
    replacing the qualname would leave the router driving a module the block no
    longer uses."""
    lin = nn.Linear(272, 544)
    holder = SimpleNamespace(ref=lin)
    ghost_parameter(lin, "weight", "complex")
    assert holder.ref is lin
    assert isinstance(lin, nn.Linear)
    assert parametrize.is_parametrized(lin, "weight")
    assert "weight" not in lin._parameters


def test_state_dict_round_trips():
    torch.manual_seed(0)
    a, b = nn.Linear(272, 544), nn.Linear(272, 544)
    ghost_parameter(a, "weight", "complex")
    ghost_parameter(b, "weight", "complex")
    with torch.no_grad():
        for p in b.parameters():
            p.add_(1.0)
    b.load_state_dict(a.state_dict())
    assert torch.allclose(a.weight, b.weight, atol=1e-6)
    keys = set(a.state_dict())
    assert "parametrizations.weight.original" in keys
    # the structure tables are derived, never stored
    assert not any("structure" in k for k in keys)


def test_gradient_reaches_only_the_real_tensor():
    lin = nn.Linear(272, 544)
    ghost_parameter(lin, "weight", "complex")
    lin(torch.randn(4, 272)).pow(2).sum().backward()
    real = lin.parametrizations.weight.original
    assert real.grad is not None and real.grad.abs().sum() > 0
    assert real.numel() == 544 * 272 // 2
