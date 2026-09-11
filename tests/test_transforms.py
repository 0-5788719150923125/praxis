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

from praxis import registry
from praxis.transforms import (
    MIN_TARGET_NUMEL,
    aligned_size,
    apply_transform,
    block_alignment,
)
from praxis.transforms.algebra import (
    ALGEBRAS,
    has_antipodal_pair,
    non_degenerate,
    structure_matrices,
    tables,
)
from praxis.transforms.alignment import align_axis
from praxis.transforms.ghost import (
    AUTO_ORDER,
    GhostExpansion,
    ghost_parameter,
    pick_algebra,
)

CONV_SHAPE = (544, 272, 3)  # abstractinator-o's ConvBlock.conv
PEER_SHAPE = (729, 272)  # PEER's default bank: gcd(729, 272) == 1


# --- the algebra ------------------------------------------------------------


@pytest.mark.parametrize("algebra", sorted(ALGEBRAS))
def test_algebra_is_non_degenerate(algebra):
    """The paper's own condition: every P_k non-singular (section 2.1). It is what
    carries the universal-approximation guarantee to ANY dimension d, which is
    why d=2 is admissible and why the cyclic family can be any size at all."""
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


def test_cyclic_algebra_is_the_group_algebra_of_z_mod_d():
    """(x.w)_k = sum_{i+j=k mod d} x_i w_j, so PERM[k][p] = (k-p) mod d, signs +1.
    Permutation matrices, so non-degenerate at every d - which is what lets d be
    chosen to DIVIDE THE TENSOR rather than the tensor chosen to suit d."""
    perm, sign = tables("cyclic3")
    assert torch.all(sign == 1)
    for k in range(3):
        for p_ in range(3):
            assert perm[k, p_].item() == (k - p_) % 3


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
    assert pick_algebra((729, 272), "auto") is None


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


# --- the walker -------------------------------------------------------------


class _Toy(nn.Module):
    """Mimics the real qualified names closely enough to exercise the profile
    regexes, with decoys each profile must NOT match. Every tensor is sized over
    MIN_TARGET_NUMEL so the floor is not what the test measures."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=100)
        self.encoder = nn.Module()
        for half in ("encoder", "decoder"):
            stack = nn.Module()
            stack.layers = nn.ModuleList()
            for _ in range(3):
                block = nn.Module()
                block.conv = nn.Conv1d(32, 64, kernel_size=3)
                block.proj = nn.Linear(64, 64, bias=False)
                stack.layers.append(block)
            setattr(self.encoder, half, stack)
        self.decoder = nn.Module()
        self.decoder.conv = nn.Conv1d(32, 64, kernel_size=3)  # conv decoy
        self.mtp = nn.Module()
        self.mtp.bank = nn.Module()
        self.mtp.bank.depths = nn.ModuleList()
        for _ in range(3):
            d = nn.Module()
            d.projection = nn.Linear(128, 64)
            d.norm = nn.Linear(64, 64)  # mtp decoy
            self.mtp.bank.depths.append(d)
        self.embeds = nn.Embedding(64, 128)
        self.bag = nn.EmbeddingBag(64, 128, mode="sum")
        self.lm_head = nn.Linear(64, 100)  # vocab-dimensioned
        self.tiny = nn.Linear(8, 8)  # under MIN_TARGET_NUMEL
        self.odd = nn.Linear(128, 63)  # indivisible on the output axis


def test_ghostify_hits_exactly_the_profiled_targets():
    model = _Toy()
    stats = apply_transform(model, "ghost_conv_complex")
    assert len(stats.targets) == 6
    assert stats.before == 6 * 64 * 32 * 3
    assert stats.after == stats.before // 2
    assert all(n.startswith("encoder.") for n, _, _, _ in stats.targets)
    assert not parametrize.is_parametrized(model.decoder.conv, "weight")
    assert not parametrize.is_parametrized(
        model.encoder.encoder.layers[0].proj, "weight"
    )


@pytest.mark.parametrize("profile", sorted(registry.namespace("transforms")))
def test_every_profile_builds_and_runs(profile):
    model = _Toy()
    stats = apply_transform(model, profile)
    assert stats.after < stats.before
    if "_mtp_" in profile:
        assert len(stats.targets) == 3
        assert all(n.endswith(".projection") for n, _, _, _ in stats.targets)
        y = model.mtp.bank.depths[0].projection(torch.randn(2, 128))
    elif "_conv_" in profile:
        assert len(stats.targets) == 6
        y = model.encoder.encoder.layers[0].conv(torch.randn(2, 32, 10))
    else:
        # Broad profiles reach every host type, not just one.
        names = [n for n, _, _, _ in stats.targets]
        assert "embeds" in names and "bag" in names
        assert any(n.endswith(".conv") for n in names)
        assert any(n.endswith(".projection") for n in names)
        y = model.embeds(torch.randint(0, 64, (2, 5)))
    y.sum().backward()


def test_broad_profile_spares_vocab_small_and_indivisible():
    model = _Toy()
    stats = apply_transform(model, "ghost_all_complex")
    names = [n for n, _, _, _ in stats.targets]
    for attr, reason in (
        ("lm_head", "vocab"),
        ("tiny", "too_small"),
        ("odd", "indivisible"),
    ):
        assert attr not in names
        assert attr in stats.missed[reason]
    assert isinstance(model.lm_head, nn.Linear)
    greedy = apply_transform(_Toy(), "ghost_all_greedy_complex")
    assert "lm_head" in [n for n, _, _, _ in greedy.targets]


def test_ghost_opaque_subtrees_are_skipped_and_reported():
    """A module whose parameters are addressed by a name captured at
    construction, or rewritten in place by an inner loop, cannot be ghosted:
    the transform RENAMES (`weight` -> `parametrizations.weight.original`) and
    DERIVES. `praxis.memory.NeuralMemory` is the real case - its fast-weight Adam
    looks its own tensors up through `self._param_names`, and ghosting it raised
    `KeyError: '0.weight'` on the first forward."""
    model = _Toy()
    model.encoder.decoder.GHOST_OPAQUE = True
    stats = apply_transform(model, "ghost_all_complex")
    names = [n for n, _, _, _ in stats.targets]
    assert not any(n.startswith("encoder.decoder.") for n in names)
    assert all(n.startswith("encoder.decoder.") for n in stats.missed["opaque"])
    assert any(n.startswith("encoder.encoder.") for n in names)


def test_ghost_opaque_is_not_merge_opaque():
    """PEER sets MERGE_OPAQUE - a claim about routing granularity - and must stay
    ghost-eligible, because its banks are the largest tensor group in the
    decoder."""
    from praxis.dense.peer import ParameterEfficientExpertRetrieval as PEER

    assert PEER.MERGE_OPAQUE is True
    assert getattr(PEER, "GHOST_OPAQUE", False) is False


def test_missed_records_names_not_just_counts():
    stats = apply_transform(_Toy(), "ghost_all_complex")
    for reason, count in stats.skipped.items():
        assert len(stats.missed[reason]) == count


def test_ghostify_none_is_a_noop_and_unknown_raises():
    model = _Toy()
    assert apply_transform(model, "none").targets == []
    assert not parametrize.is_parametrized(
        model.encoder.encoder.layers[0].conv, "weight"
    )
    with pytest.raises(ValueError):
        apply_transform(_Toy(), "no_such_profile")
    with pytest.raises(ValueError):
        apply_transform(nn.Linear(4, 4), "ghost_conv_complex")


def test_double_ghostify_is_refused_not_stacked():
    model = _Toy()
    apply_transform(model, "ghost_conv_complex")
    stats = apply_transform(model, "ghost_all_complex")
    already = stats.missed.get("already_parametrized", [])
    assert len(already) == 6 and all(n.endswith(".conv") for n in already)
    # ...and none of them were expanded a second time.
    assert not any(n in already for n, _, _, _ in stats.targets)


# --- the alignment request --------------------------------------------------


def _peer(transform_type, profile="peer_glu"):

    return registry.lookup("dense", profile)(
        SimpleNamespace(
            hidden_size=272,
            num_heads=1,
            activation="swish",
            dropout=0.0,
            transform_type=transform_type,
        )
    )


def test_peer_requests_alignment_and_that_reaches_the_banks():
    """The interface that replaced `even_keys`. At hidden_size 272 the plain round
    lands on 27 keys, giving 729 = 3^6 experts against 272 = 2^4 * 17. Coprime, so
    no d > 1 divides both axes and the largest tensor group in the model is
    unreachable. Configuring a broad ghost profile is now the whole fix: PEER asks
    what it needs and rounds to 26."""
    plain = _peer("none")
    aligned = _peer("ghost_all_complex")
    assert plain.num_experts == 729 and aligned.num_experts == 676
    assert pick_algebra(tuple(plain.down.weight.shape), "auto") is None
    assert pick_algebra(tuple(aligned.down.weight.shape), "auto") is not None


def test_site_specific_profiles_leave_other_sites_alone():
    """A profile that names one site must not resize anything else, or the run
    stops being one change off its baseline. `ghost_conv_complex` is -q, and -q's
    PEER banks have to match -o's exactly."""
    assert block_alignment(SimpleNamespace(transform_type="ghost_conv_complex")) == 1
    assert _peer("ghost_conv_complex").num_experts == _peer("none").num_experts


def test_every_unrestricted_profile_requests_alignment():
    """The guard on adding a profile. A spec that matches any name reaches the
    auto-sized modules, so it is one of the profiles they should be asking about;
    forgetting the flag would silently leave the largest banks indivisible."""
    for name, entry in registry.namespace("transforms").items():
        unrestricted = entry.spec.matches("decoder.0.ffn.down") and entry.spec.matches(
            "encoder.encoder.layers.0.conv"
        )
        assert unrestricted == entry.request_alignment, name


def test_auto_requests_the_deepest_cut():
    """`auto` walks AUTO_ORDER preferring the deepest algebra, so the request has
    to be the deepest too - an axis divisible by 4 is divisible by 2, so asking
    for the larger costs nothing and keeps quaternion reachable."""
    d = block_alignment(SimpleNamespace(transform_type="ghost_all_auto"))
    assert d == max(len(ALGEBRAS[name][0]) for name in AUTO_ORDER)
    assert _peer("ghost_all_auto").num_experts % d == 0


@pytest.mark.parametrize(
    "hidden, expected", [(272, 26), (284, 28), (512, 36), (1024, 52)]
)
def test_alignment_generalizes_across_widths(hidden, expected):
    """Derived, not tuned: the same sqrt of the same budget on a different lattice.
    These are the widths the module's own comment claims, asserted rather than
    trusted."""
    root = math.sqrt(4 * hidden * 2 / 3)
    assert align_axis(root, 2, lambda k: k**2, minimum=2) == expected


def test_alignment_measures_from_the_true_derived_value():
    """26.93 rounds to 27, and both 26 and 28 satisfy d = 2 - but 26 is nearer the
    value the budget actually produced. Rounding first and then stepping off the
    rounded base would pick 28 half the time."""
    assert align_axis(26.93, 2, lambda k: k**2, minimum=2) == 26
    assert align_axis(27.4, 2, lambda k: k**2, minimum=2) == 28
    # No request is an ordinary round, and the floor still holds.
    assert align_axis(26.93, 1) == 27
    assert align_axis(1.2, 2, minimum=4) == 4


def test_alignment_is_advisory_not_a_forced_march():
    """A request that cannot be met comes back unaligned rather than dragging the
    model somewhere far away, and the transform then reports the tensor as
    indivisible in the [GHOST] block. A missed request is a log line."""
    # An extent that is odd at every candidate: nothing satisfies d = 2.
    assert align_axis(27.0, 2, lambda k: 2 * k + 1, minimum=2) == 27
    # And a config carrying no `transform_type` at all is simply not asking.
    assert aligned_size(SimpleNamespace(), 27.0) == 27


def test_alignment_cannot_grant_the_other_axis():
    """Granting the row axis is not the same as being ghost-eligible: `d` has to
    divide the hidden axis too, and that one belongs to the config. cyclic3 would
    be satisfied by 27 keys (729 = 3^6) and still refused, because 272 = 2^4 * 17.
    Named so the request is not mistaken for a guarantee."""
    assert align_axis(26.93, 3, lambda k: k**2, minimum=2) == 27
    assert pick_algebra((729, 272), "cyclic3") is None
