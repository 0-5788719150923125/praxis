"""Ghost features: the algebra, the expansions, and the targeting pass.

The properties pinned here are the ones the experiment's interpretation rests
on. If the complex product is wrong the arm is not testing the paper's
mechanism; if the budgets are not matched the -q vs -r comparison is a budget
artifact; if the targeting pass misses a conv the run is not the experiment.
"""

import math

import pytest
import torch
import torch.nn as nn

from praxis.ghost import GHOST_REGISTRY, ghostify
from praxis.ghost.algebra import (
    ALGEBRAS,
    has_antipodal_pair,
    non_degenerate,
    structure_matrices,
    tables,
)
from praxis.ghost.expansions import EXPANSION_REGISTRY, AlgebraExpansion
from praxis.ghost.modules import GhostConv1d, GhostLinear

CONV_SHAPE = (544, 272, 3)  # abstractinator-o's ConvBlock.conv


@pytest.mark.parametrize("algebra", sorted(ALGEBRAS))
def test_algebra_is_non_degenerate(algebra):
    """The paper's own condition: every P_k non-singular (section 2.1). This is
    what carries the universal-approximation guarantee to any dimension d, and
    it is why d=2 is admissible rather than a deviation from the paper."""
    perm, sign = tables(algebra)
    assert non_degenerate(perm, sign)
    dets = torch.linalg.det(structure_matrices(perm, sign))
    assert torch.allclose(dets.abs(), torch.ones_like(dets))


@pytest.mark.parametrize("algebra", sorted(ALGEBRAS))
def test_algebra_has_no_antipodal_pair(algebra):
    """Stricter than the paper, and earned: this stack runs periodic (odd)
    activations, which commute with a global sign flip, so two blocks related by
    P_j = -P_k would collapse after the nonlinearity."""
    perm, sign = tables(algebra)
    assert not has_antipodal_pair(perm, sign)


def test_complex_expansion_reproduces_the_complex_product():
    """Re{x.w} = x0 w0 - x1 w1 and Im{x.w} = x0 w1 + x1 w0, exactly."""
    torch.manual_seed(0)
    exp = AlgebraExpansion((4, 4), algebra="complex")
    full = exp()
    real = exp.real
    x = torch.randn(4)
    for row in range(real.shape[0]):
        w0, w1 = real[row, 0], real[row, 1]
        re_block = full[row]
        im_block = full[real.shape[0] + row]
        assert torch.allclose(
            re_block[0] * x[0] + re_block[1] * x[1], x[0] * w0 - x[1] * w1, atol=1e-6
        )
        assert torch.allclose(
            im_block[0] * x[0] + im_block[1] * x[1], x[0] * w1 + x[1] * w0, atol=1e-6
        )


def test_quaternion_expansion_matches_the_multiplication_table():
    torch.manual_seed(0)
    exp = AlgebraExpansion((8, 4), algebra="quaternion")
    full, w = exp(), exp.real
    rows = w.shape[0]
    for r in range(rows):
        w0, w1, w2, w3 = w[r, 0], w[r, 1], w[r, 2], w[r, 3]
        expected = [
            (w0, -w1, -w2, -w3),
            (w1, w0, w3, -w2),
            (w2, -w3, w0, w1),
            (w3, w2, -w1, w0),
        ]
        for k, comps in enumerate(expected):
            got = full[k * rows + r, :4]
            assert torch.allclose(got, torch.stack(comps), atol=1e-6)


@pytest.mark.parametrize(
    "rule,ratio", [("complex", 0.5), ("quaternion", 0.25), ("random", 0.5)]
)
def test_algebra_budgets_are_exact(rule, ratio):
    exp = EXPANSION_REGISTRY[rule](CONV_SHAPE, "t")
    assert tuple(exp().shape) == CONV_SHAPE
    assert exp.real_numel == round(math.prod(CONV_SHAPE) * ratio)


def test_lowrank_control_is_budget_matched_and_never_favoured():
    """The rank is solved for, not set. The control must not end up with MORE
    parameters than the arm it controls, or a -q win could be a budget
    artifact rather than the structure."""
    ghost = EXPANSION_REGISTRY["complex"](CONV_SHAPE, "t")
    control = EXPANSION_REGISTRY["lowrank"](CONV_SHAPE, "t")
    assert tuple(control().shape) == CONV_SHAPE
    assert control.real_numel <= ghost.real_numel
    assert control.real_numel / ghost.real_numel > 0.99


def test_random_control_meets_the_same_conditions_as_the_algebra():
    """Otherwise the control tests degeneracy rather than arbitrariness."""
    exp = EXPANSION_REGISTRY["random"](CONV_SHAPE, "site")
    assert non_degenerate(exp.perm, exp.sign)
    assert not has_antipodal_pair(exp.perm, exp.sign)


def test_random_control_is_reproducible_across_constructions():
    """Seeded from the target's qualified name via crc32, because Python's
    hash() is salted per process and would reroll the control on restart."""
    a = EXPANSION_REGISTRY["random"](CONV_SHAPE, "encoder.encoder.layers.0.conv")
    b = EXPANSION_REGISTRY["random"](CONV_SHAPE, "encoder.encoder.layers.0.conv")
    c = EXPANSION_REGISTRY["random"](CONV_SHAPE, "encoder.encoder.layers.1.conv")
    assert torch.equal(a.perm, b.perm) and torch.equal(a.sign, b.sign)
    assert not (torch.equal(a.perm, c.perm) and torch.equal(a.sign, c.sign))


def test_expansion_rejects_shapes_that_do_not_divide():
    with pytest.raises(ValueError):
        AlgebraExpansion((7, 4), algebra="complex")
    with pytest.raises(ValueError):
        AlgebraExpansion((4, 7), algebra="complex")


@pytest.mark.parametrize("rule", sorted(EXPANSION_REGISTRY))
def test_expansions_are_differentiable(rule):
    exp = EXPANSION_REGISTRY[rule]((8, 8), "t")
    # A squared readout, not .sum(): a real element appears d times under
    # signs that can cancel, so a plain sum can hand a parameter zero gradient
    # for reasons that say nothing about differentiability.
    exp().pow(2).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in exp.parameters())


def test_ghost_conv_forward_equals_conv_with_the_expanded_weight():
    torch.manual_seed(0)
    base = nn.Conv1d(8, 16, kernel_size=3, padding=2)
    ghost = GhostConv1d(base, EXPANSION_REGISTRY["complex"], tag="t")
    x = torch.randn(2, 8, 12)
    expected = nn.functional.conv1d(x, ghost.expansion(), ghost.bias, 1, 2, 1, 1)
    assert torch.allclose(ghost(x), expected, atol=1e-6)
    assert ghost(x).shape == base(x).shape


def test_ghost_linear_forward_and_shape_preservation():
    torch.manual_seed(0)
    base = nn.Linear(8, 16)
    ghost = GhostLinear(base, EXPANSION_REGISTRY["complex"], tag="t")
    x = torch.randn(3, 8)
    assert ghost(x).shape == base(x).shape
    assert torch.allclose(ghost(x), x @ ghost.weight.T + ghost.bias, atol=1e-6)


def test_bias_stays_real():
    """Tying a [out] bias against an [out, in, k] weight saves nothing; it is
    the note's own 'combine site' mistake in miniature."""
    base = nn.Conv1d(8, 16, kernel_size=3)
    ghost = GhostConv1d(base, EXPANSION_REGISTRY["complex"], tag="t")
    assert ghost.bias is base.bias
    assert isinstance(ghost.bias, nn.Parameter)


class _Toy(nn.Module):
    """Mimics the abstractinator's qualified names closely enough to exercise
    the profile regex, including a decoy the profile must not match."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        for half in ("encoder", "decoder"):
            stack = nn.Module()
            stack.layers = nn.ModuleList()
            for _ in range(3):
                block = nn.Module()
                block.conv = nn.Conv1d(16, 32, kernel_size=3)
                block.proj = nn.Linear(16, 16, bias=False)
                stack.layers.append(block)
            setattr(self.encoder, half, stack)
        self.decoder = nn.Module()
        self.decoder.conv = nn.Conv1d(16, 32, kernel_size=3)  # decoy


def test_ghostify_hits_exactly_the_profiled_targets():
    model = _Toy()
    stats = ghostify(model, "conv_complex")
    assert len(stats.targets) == 6
    assert stats.before == 6 * 32 * 16 * 3
    assert stats.after == stats.before // 2
    for name, _, _ in stats.targets:
        assert name.startswith("encoder.")
    # The decoy and every proj survive untouched.
    assert isinstance(model.decoder.conv, nn.Conv1d)
    assert isinstance(model.encoder.encoder.layers[0].proj, nn.Linear)
    assert isinstance(model.encoder.encoder.layers[0].conv, GhostConv1d)


def test_ghostify_none_is_a_noop():
    model = _Toy()
    stats = ghostify(model, "none")
    assert stats.targets == []
    assert isinstance(model.encoder.encoder.layers[0].conv, nn.Conv1d)


def test_ghostify_raises_rather_than_silently_matching_nothing():
    with pytest.raises(ValueError):
        ghostify(nn.Linear(4, 4), "conv_complex")
    with pytest.raises(ValueError):
        ghostify(_Toy(), "no_such_profile")


@pytest.mark.parametrize("profile", sorted(GHOST_REGISTRY))
def test_every_profile_builds_and_runs(profile):
    model = _Toy()
    ghostify(model, profile)
    x = torch.randn(2, 16, 10)
    out = model.encoder.encoder.layers[0].conv(x)
    assert out.shape[1] == 32
    out.sum().backward()


def _reference_expand(real, perm, sign, d, out, in_, tail):
    """Independent, deliberately naive implementation of the construction.

    ``expanded[k*rows + r, g*d + p] = SIGN[k][p] * real[r, g*d + PERM[k][p]]``,
    written as loops so it cannot share a bug with the vectorized paths.
    """
    rows = out // d
    full = torch.zeros(out, in_, *tail)
    for k in range(d):
        for r in range(rows):
            for g in range(in_ // d):
                for pos in range(d):
                    full[k * rows + r, g * d + pos] = (
                        sign[k, pos] * real[r, g * d + perm[k, pos]]
                    )
    return full


@pytest.mark.parametrize(
    "algebra,shape",
    [
        ("complex", (8, 6, 2)),
        ("quaternion", (8, 8, 2)),
        ("complex", (6, 4)),
        ("quaternion", (8, 4)),
    ],
)
def test_expansion_matches_the_naive_reference(algebra, shape):
    torch.manual_seed(0)
    exp = AlgebraExpansion(shape, algebra=algebra)
    out, in_, *tail = shape
    ref = _reference_expand(exp.real, exp.perm, exp.sign, exp.d, out, in_, tuple(tail))
    assert torch.allclose(exp(), ref, atol=1e-6)


@pytest.mark.parametrize("algebra", sorted(ALGEBRAS))
def test_structure_buffer_matches_the_perm_sign_tables(algebra):
    """``forward`` contracts against ``structure``, while the non-degeneracy and
    antipodal guards are asserted against ``perm``/``sign``. If those two ever
    disagree, the guards are vetting an algebra the model does not use."""
    exp = AlgebraExpansion((8, 8), algebra=algebra)
    assert torch.allclose(
        exp.structure.double(), structure_matrices(exp.perm, exp.sign)
    )


def test_random_control_expands_by_its_own_drawn_permutation():
    """The control draws its own permutation, so it must expand by that draw
    rather than by the algebra's."""
    exp = EXPANSION_REGISTRY["random"]((8, 6, 2), "site")
    rows, in_ = exp.real.shape[0], exp.real.shape[1]
    ref = _reference_expand(exp.real, exp.perm, exp.sign, exp.d, 8, 6, (2,))
    assert torch.allclose(exp(), ref, atol=1e-6)
