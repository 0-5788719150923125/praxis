"""Ghost features: the algebra, the expansions, and the targeting pass.

The properties pinned here are the ones the experiment's interpretation rests
on. If the complex product is wrong the arm is not testing the paper's
mechanism; if the budgets are not matched the -q vs -r comparison is a budget
artifact; if the targeting pass misses a conv the run is not the experiment.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from praxis.ghost import GHOST_REGISTRY, MIN_TARGET_NUMEL, ghostify
from praxis.ghost.algebra import (
    ALGEBRAS,
    has_antipodal_pair,
    non_degenerate,
    structure_matrices,
    tables,
)
from praxis.ghost.expansions import (
    EXPANSION_REGISTRY,
    AlgebraExpansion,
    Expansion,
)
from praxis.ghost.modules import GhostConv1d, GhostEmbedding, GhostLinear

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
    """Mimics the real qualified names closely enough to exercise the profile
    regexes, with decoys each profile must NOT match. Every tensor is sized over
    ``MIN_TARGET_NUMEL`` so the floor is not what the test is measuring."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=100)
        self.encoder = nn.Module()
        for half in ("encoder", "decoder"):
            stack = nn.Module()
            stack.layers = nn.ModuleList()
            for _ in range(3):
                block = nn.Module()
                block.conv = nn.Conv1d(32, 64, kernel_size=3)  # [64, 32, 3]
                block.proj = nn.Linear(64, 64, bias=False)  # [64, 64]
                stack.layers.append(block)
            setattr(self.encoder, half, stack)
        self.decoder = nn.Module()
        self.decoder.conv = nn.Conv1d(32, 64, kernel_size=3)  # conv decoy
        self.mtp = nn.Module()
        self.mtp.bank = nn.Module()
        self.mtp.bank.depths = nn.ModuleList()
        for _ in range(3):
            d = nn.Module()
            d.projection = nn.Linear(128, 64)  # [64, 128]
            d.norm = nn.Linear(64, 64)  # mtp decoy
            self.mtp.bank.depths.append(d)
        self.embeds = nn.Embedding(64, 128)  # [64, 128]
        self.lm_head = nn.Linear(64, 100)  # [100, 64], vocab-dimensioned
        self.tiny = nn.Linear(8, 8)  # under MIN_TARGET_NUMEL


def test_ghostify_hits_exactly_the_profiled_targets():
    model = _Toy()
    stats = ghostify(model, "conv_complex")
    assert len(stats.targets) == 6
    assert stats.before == 6 * 64 * 32 * 3
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
    stats = ghostify(model, profile)
    assert stats.after < stats.before
    if profile.startswith("mtp_"):
        # Exactly the per-depth projections, never their norm siblings.
        assert len(stats.targets) == 3
        assert all(n.endswith(".projection") for n, _, _ in stats.targets)
        assert isinstance(model.mtp.bank.depths[0].norm, nn.Linear)
        y = model.mtp.bank.depths[0].projection(torch.randn(2, 128))
    elif profile.startswith("conv_"):
        assert len(stats.targets) == 6
        y = model.encoder.encoder.layers[0].conv(torch.randn(2, 32, 10))
    else:
        # Broad profiles reach every kind of wrapper, not just one.
        kinds = {type(model.get_submodule(n)).__name__ for n, _, _ in stats.targets}
        assert {"GhostConv1d", "GhostLinear", "GhostEmbedding"} <= kinds
        y = model.embeds(torch.randint(0, 64, (2, 5)))
    y.sum().backward()


def test_broad_profile_spares_the_vocab_tensors_and_the_small_ones():
    model = _Toy()
    stats = ghostify(model, "all_complex")
    names = [n for n, _, _ in stats.targets]
    assert "lm_head" not in names and "lm_head" in stats.missed.get("vocab", [])
    assert "tiny" not in names and "tiny" in stats.missed.get("too_small", [])
    assert isinstance(model.lm_head, nn.Linear)
    # ...and the greedy profile does not spare the readout.
    greedy = ghostify(_Toy(), "all_greedy_complex")
    assert "lm_head" in [n for n, _, _ in greedy.targets]


def test_broad_profile_reaches_more_than_the_narrow_ones():
    narrow = ghostify(_Toy(), "conv_complex")
    broad = ghostify(_Toy(), "all_complex")
    assert len(broad.targets) > len(narrow.targets)
    assert broad.before > narrow.before


def test_missed_records_names_not_just_counts():
    """A broad profile that silently covers a third of what you think it covers
    is worse than one that covers nothing."""
    stats = ghostify(_Toy(), "all_complex")
    for reason, count in stats.skipped.items():
        assert len(stats.missed[reason]) == count


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


@pytest.mark.parametrize("rule", sorted(EXPANSION_REGISTRY))
@pytest.mark.parametrize("shape", [CONV_SHAPE, (256, 128, 3), (272, 272)])
def test_init_scale_matches_the_replaced_module(rule, shape):
    """The EXPANDED weight must start at the scale the original module would.

    This is the regression for the bug that invalidated the first -r run: the
    lowrank control init'd at 0.045x the correct scale, so six conv layers began
    22x too quiet and the arm was measuring an init as well as a mechanism. An
    arm that starts somewhere else is not a control.
    """
    torch.manual_seed(0)
    out, in_, *tail = shape
    if tail:
        ref = nn.Conv1d(in_, out, kernel_size=tail[0]).weight
    else:
        ref = nn.Linear(in_, out).weight
    got = EXPANSION_REGISTRY[rule](shape, "t")()
    ratio = (got.std() / ref.std()).item()
    assert (
        0.85 < ratio < 1.15
    ), f"{rule} at {shape}: init scale {ratio:.3f}x the original"


@pytest.mark.parametrize("rank_shape", [(544, 272, 3), (128, 64, 3), (64, 64)])
def test_lowrank_init_is_solved_not_fitted(rank_shape):
    """``rank * Var(U) * Var(V) == GAIN_SQ / fan`` must hold exactly, at any
    rank, so the scale cannot drift when the profile moves to another shape."""
    exp = EXPANSION_REGISTRY["lowrank"](rank_shape, "t")
    out, in_, *tail = rank_shape
    fan = in_ * math.prod(tail) if tail else in_
    predicted = exp.rank * exp.u.var().item() * exp.v.var().item()
    target = Expansion.GAIN_SQ / fan
    assert abs(predicted - target) / target < 0.15


@pytest.mark.parametrize(
    "rule,expected_full_rank", [("complex", True), ("quaternion", True),
                                ("random", True), ("lowrank", False)]
)
def test_algebra_expansions_are_full_rank_and_lowrank_is_not(rule, expected_full_rank):
    """The fact that retired the first -r arm, pinned so it is not re-litigated.

    ``P_k`` acts on the INPUT axis, so the d expanded blocks are not linear
    combinations of the ``out // d`` real rows - each applies a different
    input-space transform, and the stack comes out FULL rank. A parameter-matched
    low-rank factor mixes the OUTPUT axis and is capped at
    ``params / (out + fan)``, which at -o's conv shape is 163 of 544. Matched
    budget, different rank class: not a control.
    """
    exp = EXPANSION_REGISTRY[rule](CONV_SHAPE, "t")
    W = exp().detach().reshape(CONV_SHAPE[0], -1)
    rank = torch.linalg.matrix_rank(W).item()
    if expected_full_rank:
        assert rank == CONV_SHAPE[0]
    else:
        assert rank == exp.rank < CONV_SHAPE[0]


def test_ghost_opaque_subtrees_are_skipped_and_reported():
    """A module whose parameters are addressed by a name captured at
    construction, or rewritten in place by an inner loop, cannot be ghosted:
    ghosting RENAMES (`weight` -> `expansion.real`) and DERIVES (nothing to write
    back to). `praxis.memory.NeuralMemory` is the real case - its fast-weight
    Adam looks its own tensors up through `self._param_names`, and ghosting it
    raised `KeyError: '0.weight'` on the first forward.

    Separate from MERGE_OPAQUE on purpose: a module can be opaque to the
    parameter-merging routers and open to this transform, and PEER is exactly
    that module.
    """
    model = _Toy()
    model.encoder.decoder.GHOST_OPAQUE = True
    stats = ghostify(model, "all_complex")
    names = [n for n, _, _ in stats.targets]
    assert not any(n.startswith("encoder.decoder.") for n in names)
    assert stats.skipped.get("opaque", 0) >= 3
    assert all(n.startswith("encoder.decoder.") for n in stats.missed["opaque"])
    # the sibling stack is untouched by the exclusion
    assert any(n.startswith("encoder.encoder.") for n in names)


def test_ghost_opaque_is_not_merge_opaque():
    """PEER sets MERGE_OPAQUE and must stay ghost-eligible."""
    from praxis.dense.peer import ParameterEfficientExpertRetrieval as PEER

    assert PEER.MERGE_OPAQUE is True
    assert getattr(PEER, "GHOST_OPAQUE", False) is False
