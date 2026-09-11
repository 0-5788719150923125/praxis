"""praxis/classifiers/parallel.py: ParallelClassifier and SurgicalParallelClassifier.

Covers the gate (normalization, gradient, metrics, level repulsion), the stem
and its blueprint repr, the prismatic8 profile, ``_pcgrad``, the arm Jacobian
measurement every ParallelClassifier reports, and the per-arm training intervention
prismatic9 adds on top of it.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from praxis import registry
from praxis.classifiers.base import BaseClassifier
from praxis.classifiers.harmonic import HarmonicClassifier
from praxis.classifiers.parallel import (
    ARM_CONFLICT_INTERVAL,
    ParallelClassifier,
    SurgicalParallelClassifier,
    _pcgrad,
)
from praxis.losses import Objectives
from praxis.losses.cross_entropy import CrossEntropyLoss
from praxis.losses.halo import HALOLoss
from praxis.metrics.descriptions import get_metric_descriptions
from tests.stubs import Cfg, Enc


def build(name):
    torch.manual_seed(0)
    return registry.lookup("classifiers", name)(Cfg(), encoder=Enc())


def _batch(b=2, t=5):
    torch.manual_seed(1)
    x = torch.randn(b, t, Cfg.hidden_size, requires_grad=True)
    return x, torch.randint(0, Cfg.vocab_size, (b, t))


def _objectives(main=None):
    """The loss container a classifier is handed, as modeling.py builds it.

    Every arm's row comes out of here: ``arm_ce`` for the plain arms, the
    main criterion's geometry for the HALO one. Omitting ``main`` is the
    no-criterion case - the HALO arm has nothing to score with.
    """
    terms = Objectives()
    if main is not None:
        terms.register("main", main)
    terms.register("arm_ce", CrossEntropyLoss())
    return terms


def _halo_objectives():
    return _objectives(HALOLoss(vocab_size=Cfg.vocab_size))


# ── The gate ───────────────────────────────────────────────────────────────


def _parallel(n=2):
    torch.manual_seed(0)
    field = partial(HarmonicClassifier, amp_modulation="learned", build_scorer=False)
    return ParallelClassifier(Cfg(), branches=[field for _ in range(n)])


def test_transform_preserves_shape_and_gate_normalizes():
    classifier = _parallel()
    x = torch.randn(2, 8, Cfg.hidden_size)
    assert classifier.transform(x).shape == x.shape
    assert classifier._gate_mean is not None and len(classifier._gate_mean) == 2
    assert abs(float(classifier._gate_mean.sum()) - 1) < 1e-5


def test_gate_is_learned_and_receives_gradient():
    classifier = _parallel()
    classifier.transform(torch.randn(2, 8, Cfg.hidden_size)).sum().backward()
    assert classifier.gate.weight.grad is not None
    assert classifier.gate.weight.grad.abs().sum() > 0


def test_training_metrics_namespaced_with_gate():
    classifier = build("prismatic")
    logits = classifier(torch.randn(2, 8, Cfg.hidden_size))  # populates the gate stats
    assert logits.shape == (2, 8, Cfg.vocab_size)
    m = classifier.training_metrics()
    assert {"gate_weight_0", "gate_weight_1", "gate_entropy"} <= set(m)
    assert any(k.startswith("p0_harmonic") for k in m)
    assert any(k.startswith("p1_harmonic") for k in m)


def test_prismatic_descriptions_namespaced_and_attributed():
    classifier = build("prismatic")
    stub = SimpleNamespace(classifier=classifier, reg=[], tasker=None, encoder=False)
    descs = get_metric_descriptions(stub)

    for i in (0, 1):
        key = f"p{i}_harmonic_amplitudes_norm"
        assert key in descs, key
        assert descs[key]["caller"] == "HarmonicField"
        assert descs[key]["chart"]["title"].endswith(f"#{i}")

    assert descs["gate_entropy"]["caller"] == "ParallelClassifier"
    assert descs["gate_weight_0"]["chart"]["series_group"] == "parallel_gate"


# ── Gate level repulsion ───────────────────────────────────────────────────
#
# The gate's mean per-branch weights should settle at DISTINCT tiers (e.g.
# 70/20/10) rather than degenerate ties (70/15/15). ``gate_repulsion`` (bound by
# the prismatic3_repel profile) adds a pairwise log-gap penalty that repels
# equal weights apart. Off by default; the min-gap diagnostic is always charted.


class IdBranch(BaseClassifier):
    """Identity branch - the gate is what these tests exercise."""

    def transform(self, h):
        return h

    def forward(self, h, **k):
        return h

    @property
    def scorer(self):
        return None


def _repelled(lam, n=3):
    cfg = SimpleNamespace(hidden_size=8, vocab_size=8)
    return ParallelClassifier(
        cfg, encoder=None, branches=[IdBranch for _ in range(n)], gate_repulsion=lam
    )


def test_repulsion_off_by_default_but_min_gap_is_charted():
    h = _repelled(lam=0.0).train()
    h.transform(torch.randn(4, 16, 8))
    assert "gate_repulsion" not in h.aux_losses()
    tm = h.training_metrics()
    assert "gate_min_gap" in tm and "gate_entropy" in tm
    assert all(f"gate_weight_{i}" in tm for i in range(3))


def test_aux_loss_is_prescaled_by_lambda():
    x = torch.randn(4, 16, 8)
    torch.manual_seed(0)
    h1 = _repelled(lam=1.0).train()
    torch.manual_seed(0)
    h2 = _repelled(lam=3.0).train()
    # Identical gate init (same seed) -> identical repulsion energy, scaled by lam.
    h1.transform(x)
    h2.transform(x)
    r1 = float(h1.aux_losses()["gate_repulsion"].detach())
    r2 = float(h2.aux_losses()["gate_repulsion"].detach())
    assert abs(r2 - 3.0 * r1) < 1e-4


def test_repulsion_breaks_a_tie_into_distinct_tiers():
    """Start from a near-degenerate gate (all branches ~equal) and optimize the
    repulsion alone: the mean weights must separate into distinct tiers."""
    torch.manual_seed(0)
    h = _repelled(lam=1.0, n=3).train()
    # Near-tie start: tiny gate weights -> softmax ~ uniform (min_gap ~ 0).
    with torch.no_grad():
        h.gate.weight.mul_(0.01)
    x = torch.randn(8, 32, 8)

    h.transform(x)
    start_gap = h._gate_min_gap
    assert start_gap < 0.02  # genuinely tied to begin with

    opt = torch.optim.SGD(h.gate.parameters(), lr=0.5)
    for _ in range(300):
        opt.zero_grad()
        h.transform(x)
        h.aux_losses()["gate_repulsion"].backward()
        opt.step()

    h.transform(x)
    end_gap = h._gate_min_gap
    weights = sorted(h._gate_mean.tolist(), reverse=True)
    assert end_gap > start_gap + 0.1, (start_gap, end_gap)
    # All three remain distinct (no two within the floor of each other).
    assert weights[0] - weights[1] > 0.02 and weights[1] - weights[2] > 0.02


# ── Stem and blueprint ─────────────────────────────────────────────────────

_PRISMATIC8_REPR = (
    "Parallel(stem=HarmonicField, arms=[CrystalClassifier, LinearClassifier, "
    "HaloClassifier(reads_trunk=True)])"
)


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "prismatic",
            "Parallel(arms=[Sequential(HarmonicField), "
            "Sequential(HarmonicField, CrystalClassifier)])",
        ),
        ("prismatic8", _PRISMATIC8_REPR),
        ("prismatic9", "Surgical" + _PRISMATIC8_REPR),
    ],
)
def test_blueprint_repr(name, expected):
    """Keyword style, like every other module in the blueprint: the stem is a
    keyword rather than an arrow, each arm names its readout (geometric,
    direct, hyperspherical), and the HALO arm says it branches above the stem."""
    assert repr(build(name)) == expected


def test_the_stem_does_not_feed_every_arm():
    """An arm with ``reads_trunk`` branches above the stem and scores the raw
    trunk hidden states - HALOLoss scores those same features, and a transform
    in front would train one feature space and score another."""
    classifier = build("prismatic8")
    trunk = torch.randn(2, 6, Cfg.hidden_size)
    stemmed = classifier._stem_out(trunk)
    reads = {
        type(b).__name__: classifier._branch_input(b, stemmed, trunk) is stemmed
        for b in classifier.branches
    }
    assert reads["CrystalClassifier"] and reads["LinearClassifier"]
    assert not reads["HaloClassifier"], "HALO must score trunk features, not the stem"


def test_prismatic3_pure_arm_is_identity_at_init():
    classifier = build("prismatic3")
    x = torch.randn(2, 6, Cfg.hidden_size)
    assert classifier(x).shape == (2, 6, Cfg.vocab_size)
    torch.testing.assert_close(classifier.branches[2].transform(x), x)


# ── prismatic8: three fixed arms over one stem ─────────────────────────────


def test_prismatic8_reads_causally():
    """A single crystal reads position t from position t, so the speculative
    decoder never re-encodes a row per candidate."""
    classifier = build("prismatic8")
    assert classifier.branches[0].causal_readout is True
    assert classifier.causal_readout is True


def test_logits_shape_and_mixture_is_normalized():
    classifier = build("prismatic8").eval()
    x = torch.randn(2, 5, Cfg.hidden_size)
    with torch.no_grad():
        out = classifier(x)
    assert out.shape == (2, 5, Cfg.vocab_size)
    # _gate_combine_logits emits a log-prob mixture: rows exponentiate to 1.
    assert torch.allclose(out.exp().sum(-1), torch.ones(2, 5), atol=1e-4)


def test_geometry_is_one_table_in_its_own_frame():
    """A single PCA card for the single crystal."""
    snaps = build("prismatic8").dashboard_snapshots()
    pca = [k for k in snaps if "crystal_centers_pca" in k]
    assert pca == ["p0_crystal_centers_pca"], pca


def test_gradient_reaches_the_crystal_centers():
    classifier = build("prismatic8")
    classifier(torch.randn(2, 5, Cfg.hidden_size)).sum().backward()
    centers = classifier.branches[0].scorer.centers
    assert centers.grad is not None
    assert torch.isfinite(centers.grad).all()
    assert centers.grad.abs().sum() > 0


@pytest.mark.parametrize("n_pos", [1, 7])
def test_prefix_invariance(n_pos):
    """Reading position t out of a longer row equals running the prefix that
    ends at t."""
    classifier = build("prismatic8").eval()
    torch.manual_seed(1)
    x = torch.randn(1, 8, Cfg.hidden_size)
    with torch.no_grad():
        full = classifier(x)[0, n_pos]
        prefix = classifier(x[:, : n_pos + 1])[0, n_pos]
    assert torch.allclose(full, prefix, atol=1e-5)


# ── PCGrad ─────────────────────────────────────────────────────────────────


def test_pcgrad_is_the_plain_sum_when_nothing_conflicts():
    """Its designed no-op. This is what makes it safe as a default rather than
    a bet: it can only act where there is something to act on."""
    g = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0])]
    assert torch.allclose(_pcgrad(g), sum(g))


def test_pcgrad_removes_the_conflicting_component():
    """With two rows there is no order to depend on, so this is Yu et al.
    unmodified: each row loses its projection onto the other."""
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0])
    out = _pcgrad([a, b])
    a_p = a - (a @ b) / (b @ b) * b
    b_p = b - (b @ a) / (a @ a) * a
    assert torch.allclose(out, a_p + b_p, atol=1e-6)
    # The plain sum would have cancelled the first coordinate to zero; PCGrad
    # is what stops one objective silently erasing the other.
    assert torch.allclose(a + b, torch.tensor([0.0, 1.0]))
    assert out[0] > 0


def test_pcgrad_is_order_independent():
    """The deliberate deviation from Yu et al., who project sequentially and
    randomize task order to unbias the resulting order-dependence. Measuring
    every projection against the original gradients keeps the RNG out of the
    training path."""
    torch.manual_seed(3)
    g = [torch.randn(2, 5, 7) for _ in range(4)]
    out = _pcgrad(g)
    assert out.shape == (2, 5, 7)
    assert torch.allclose(out, _pcgrad(list(reversed(g))), atol=1e-6)


# ── The arm Jacobian, which every ParallelClassifier measures ────────────────────


def test_arm_conflict_reports_a_full_jacobian_on_a_plain_parallel_classifier():
    """Measurement is universal: prismatic8 reports its arm Jacobian and its
    override shares without any change to how it trains."""
    classifier = build("prismatic8").train()
    x, y = _batch()
    m = classifier.arm_conflict(x, y, _halo_objectives())
    # ALL THREE arms are rows. The HALO arm's row is its own geometric
    # objective, not an invented CE - a Jacobian row is any objective's
    # gradient w.r.t. the shared representation and need not be a CE.
    pairs = {k for k in m if k.startswith("arm_cos_") and k != "arm_cos_min"}
    assert pairs == {"arm_cos_01", "arm_cos_02", "arm_cos_12"}
    assert m["arm_cos_min"] == min(m[k] for k in pairs)
    for i in range(3):
        assert 0.0 <= m[f"arm_grad_share_{i}"] <= 1.0
        assert 0.0 <= m[f"arm_override_{i}"] <= 1.0
        assert 0.0 <= m[f"arm_override_pcg_{i}"] <= 1.0
    assert max(m[f"arm_grad_share_{i}"] for i in range(3)) == pytest.approx(1.0)


def test_arm_conflict_is_sampled_and_holds_between_samples():
    classifier = build("prismatic8").train()
    x, y = _batch()
    terms = _objectives()
    first = dict(classifier.arm_conflict(x, y, terms))
    assert first
    for _ in range(ARM_CONFLICT_INTERVAL - 1):
        assert classifier.arm_conflict(x, y, terms) == first
    assert classifier._arm_step == ARM_CONFLICT_INTERVAL


def test_arm_conflict_is_inert_at_eval_and_without_labels():
    classifier = build("prismatic8")
    x, y = _batch()
    assert classifier.eval().arm_conflict(x, y, _objectives()) == {}
    assert classifier.train().arm_conflict(x, None, _objectives()) == {}


def test_arm_conflict_does_not_disturb_the_training_graph():
    """The diagnostic runs on a detached leaf, so the real backward that
    follows must be unaffected."""
    classifier = build("prismatic8").train()
    x, y = _batch()
    classifier.arm_conflict(x, y, _objectives())
    classifier(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_the_arm_row_uses_the_registered_cross_entropy():
    """Each arm's Jacobian row is an objective like any other, so it comes out
    of the container instead of a bare functional call - and a container
    without one is a hard failure, never a silent fallback."""

    class Doubled(CrossEntropyLoss):
        def forward(self, *args, **kwargs):
            return 2.0 * super().forward(*args, **kwargs)

    classifier = build("prismatic8").train()
    assert isinstance(classifier.objectives()["arm_ce"], CrossEntropyLoss)
    arm = classifier.branches[0]
    x, y = _batch()
    terms = Objectives()
    with pytest.raises(KeyError, match="arm_ce"):
        arm.arm_loss(x, y, terms)
    terms.register("arm_ce", Doubled())
    expected = 2.0 * CrossEntropyLoss()(logits=arm(x).float(), labels=y)
    assert torch.allclose(arm.arm_loss(x, y, terms), expected)


# ── Override: whether magnitude actually bit ───────────────────────────────


def test_override_is_zero_when_arms_agree():
    """Rows pointing the same way cannot overrule each other whatever their
    relative sizes: magnitude alone is not harm."""
    big = torch.tensor([20.0, 20.0, 20.0])
    small = torch.tensor([1.0, 1.0, 1.0])
    out = ParallelClassifier._override_shares(
        [small, big], [float(small.norm()), float(big.norm())], [0, 1]
    )
    assert out["arm_override_0"] == pytest.approx(0.0)
    assert out["arm_override_1"] == pytest.approx(0.0)


def test_override_catches_a_loud_row_reversing_a_quiet_one():
    """A 20:1 magnitude ratio with opposed signs: the plain sum points the
    wrong way for the quiet arm on every coordinate."""
    quiet = torch.tensor([1.0, 1.0, 1.0])
    loud = torch.tensor([-20.0, -20.0, -20.0])
    out = ParallelClassifier._override_shares(
        [quiet, loud], [float(quiet.norm()), float(loud.norm())], [0, 1]
    )
    assert out["arm_override_0"] == pytest.approx(1.0)
    assert out["arm_override_1"] == pytest.approx(0.0)
    # And PCGrad is what recovers it: fully opposed rows cancel to zero rather
    # than letting the loud one dictate.
    assert out["arm_override_pcg_0"] == pytest.approx(0.0)


def test_override_is_mass_weighted_not_counted():
    """A flipped coordinate the arm barely cared about is not an override."""
    # Arm 0 cares almost entirely about coord 0, which survives; coord 1 flips.
    a = torch.tensor([100.0, 0.01])
    b = torch.tensor([1.0, -50.0])
    out = ParallelClassifier._override_shares(
        [a, b], [float(a.norm()), float(b.norm())], [0, 1]
    )
    assert out["arm_override_0"] < 0.001, "a counted metric would report 50%"


def test_equalizing_rows_gives_a_drowned_arm_a_vote_back():
    """A 20x row that is merely ORTHOGONAL to its neighbours leaves PCGrad
    nothing to project, and 0.5 override is the coin-flip null for a small row
    against a dominant one - it means no vote, not opposition."""
    torch.manual_seed(7)
    quiet = torch.randn(4096) * 0.05
    other = torch.randn(4096) * 0.05
    loud = torch.randn(4096) * 1.0  # ~20x, and orthogonal by construction
    rows = [quiet, other, loud]
    norms = [float(g.norm()) for g in rows]
    m = ParallelClassifier._override_shares(rows, norms, [0, 1, 2])

    # Drowned: the quiet rows sit at the coin-flip null under the plain sum...
    assert 0.40 < m["arm_override_0"] < 0.60
    # ...and PCGrad cannot help, because there is no conflict to project.
    assert abs(m["arm_override_pcg_0"] - m["arm_override_0"]) < 0.05
    # Equalizing rows is what restores their say.
    assert m["arm_override_eq_0"] < 0.30
    assert m["arm_override_eq_1"] < 0.30
    # And the loud row pays for it: it stops being the only voice.
    assert m["arm_override_eq_2"] > m["arm_override_2"]


# ── SurgicalParallelClassifier (prismatic9): per-arm objectives, PCGrad trunk ────
#
# The mixture of softmaxes stays the way predictions are COMBINED. What changes
# is that it no longer decides how much each arm is TRAINED: under the mixture,
# cross-entropy reaches arm i scaled by its posterior responsibility, so an arm
# the gate stops trusting stops receiving gradient. Here each arm gets its own
# objective and the trunk receives one PCGrad-combined gradient over them.


def test_prismatic8_emits_no_arm_objectives():
    classifier = build("prismatic8").train()
    x, y = _batch()
    assert classifier.arm_objectives(x, y, _halo_objectives()) == {}
    assert classifier.arm_surgery is False


def test_prismatic9_emits_one_loss_per_arm_plus_the_surrogate():
    classifier = build("prismatic9").train()
    assert isinstance(classifier, SurgicalParallelClassifier)
    assert classifier.equalize_rows is True
    x, y = _batch()
    out = classifier.arm_objectives(x, y, _halo_objectives())
    # Every arm is a row, HALO included, so nothing reaches the trunk outside
    # the arbitration.
    assert set(out) == {"arm0_loss", "arm1_loss", "arm2_loss", "arm_surgery"}


def test_surgery_refuses_rather_than_shipping_a_partial_jacobian():
    """An objective that reaches the shared representation but sits OUTSIDE the
    arbitration routes around the very thing the surgery does. With no
    criterion the HALO arm cannot supply its row, so the whole intervention
    stands down instead of arbitrating two rows out of three."""
    classifier = build("prismatic9").train()
    x, y = _batch()
    assert classifier.arm_objectives(x, y, _objectives()) == {}
    assert classifier._arm_gap is True
    assert classifier.arm_objectives(x, y, _halo_objectives())


def test_the_gate_cannot_bypass_the_surgery():
    """The gate's CE would otherwise be a fourth uncorrected path into the
    trunk - the same defect as an excluded arm, just quieter."""
    nine, eight = build("prismatic9").train(), build("prismatic8").train()
    assert nine.detach_gate_input is True
    assert eight.detach_gate_input is False
    x, _ = _batch()
    # Under prismatic9 the blended logits carry no trunk gradient at all: every
    # arm is detached in the blend and the gate reads a detached trunk.
    g = torch.autograd.grad(nine(x).sum(), x, allow_unused=True)[0]
    assert g is None or g.abs().sum() == 0
    # prismatic8 keeps the ordinary path.
    g8 = torch.autograd.grad(eight(x).sum(), x, allow_unused=True)[0]
    assert g8 is not None and g8.abs().sum() > 0


def test_halo_row_is_its_own_objective_not_a_cross_entropy():
    classifier = build("prismatic9").train()
    x, y = _batch()
    crit = _halo_objectives()
    out = classifier.arm_objectives(x, y, crit)
    halo_arm = classifier.branches[2]
    direct = halo_arm.arm_loss(x.detach().requires_grad_(True), y, crit)
    assert torch.allclose(out["arm2_loss"], direct, atol=1e-5)
    # And it is NOT what a CE on the arm would give.
    logits = halo_arm(x).reshape(-1, Cfg.vocab_size).float()
    ce = torch.nn.functional.cross_entropy(logits, y.reshape(-1))
    assert not torch.allclose(direct, ce, atol=1e-3)


def test_the_surrogate_contributes_no_value_to_the_loss():
    """Its magnitude is arbitrary - it exists only to carry a gradient - so it
    must not land in the loss curve people read."""
    classifier = build("prismatic9").train()
    x, y = _batch()
    out = classifier.arm_objectives(x, y, _halo_objectives())
    assert float(out["arm_surgery"].detach()) == pytest.approx(0.0, abs=1e-9)
    (g,) = torch.autograd.grad(out["arm_surgery"], x, retain_graph=True)
    assert g.abs().sum() > 0, "value-neutral must not mean gradient-neutral"


def test_the_surrogate_delivers_exactly_the_pcgrad_gradient_to_the_trunk():
    """``d/dz (z * g_hat).sum() == g_hat``. This is the whole mechanism: the
    trunk receives the combined gradient without a second backward through it."""
    classifier = build("prismatic9").train()
    classifier.equalize_rows = False  # the identity is over the raw PCGrad result
    x, y = _batch()
    out = classifier.arm_objectives(x, y, _halo_objectives())
    (g,) = torch.autograd.grad(out["arm_surgery"], x, retain_graph=True)

    # Recompute the expected combination independently.
    z = classifier._stem_out(x)
    z_d = z.detach().requires_grad_(True)
    trunk_d = x.detach().requires_grad_(True)
    grads = []
    crit = _halo_objectives()
    for b in classifier.branches:
        inp = trunk_d if getattr(b, "reads_trunk", False) else z_d
        loss = b.arm_loss(inp, y, crit)
        grads.append(torch.autograd.grad(loss, inp, retain_graph=True)[0])
    expected = _pcgrad(grads)
    # The stem is dim-preserving and differentiable, so the trunk gradient is
    # g_hat pushed back through it; check the stem-level identity directly.
    (gz,) = torch.autograd.grad((z * expected.detach()).sum(), x, retain_graph=True)
    assert torch.allclose(g, gz, atol=1e-5)


def test_solo_ce_trains_the_arm_but_never_the_trunk():
    """The arms run on a detached leaf, which is what makes this affordable:
    each backward crosses one small arm, never the trunk."""
    classifier = build("prismatic9").train()
    x, y = _batch()
    out = classifier.arm_objectives(x, y, _halo_objectives())
    (g,) = torch.autograd.grad(
        out["arm0_loss"], x, allow_unused=True, retain_graph=True
    )
    assert g is None, "an arm's solo CE must not reach the trunk"

    crystal = classifier.branches[0]
    params = [p for p in crystal.parameters() if p.requires_grad]
    gs = torch.autograd.grad(
        out["arm0_loss"], params, allow_unused=True, retain_graph=True
    )
    assert any(t is not None and t.abs().sum() > 0 for t in gs)


def test_a_starved_arm_still_gets_full_gradient():
    """An arm's training signal does not pass through the gate at all, so gate
    collapse cannot starve it."""
    classifier = build("prismatic9").train()
    x, y = _batch()
    crit = _halo_objectives()
    before = classifier.arm_objectives(x, y, crit)
    with torch.no_grad():
        classifier.gate.weight.mul_(0).add_(
            torch.randn_like(classifier.gate.weight) * 100
        )
    after = classifier.arm_objectives(x, y, crit)
    # Wrecking the gate leaves every arm's own objective bit-identical.
    for k in ("arm0_loss", "arm1_loss", "arm2_loss"):
        assert torch.equal(before[k], after[k]), k

    starved = [p for p in classifier.branches[1].parameters() if p.requires_grad]
    gs = torch.autograd.grad(after["arm1_loss"], starved, allow_unused=True)
    assert any(t is not None and t.abs().sum() > 0 for t in gs)


def test_forward_and_inference_are_unchanged_vs_prismatic8():
    """Identical construction, identical predictions: prismatic9 changes
    training only. detach_in_blend is a no-op outside training."""
    nine, eight = build("prismatic9").eval(), build("prismatic8").eval()
    x, _ = _batch()
    with torch.no_grad():
        assert torch.allclose(nine(x), eight(x), atol=1e-6)


def test_metrics_and_descriptions_line_up():
    classifier = build("prismatic9").train()
    x, y = _batch()
    crit = _halo_objectives()
    classifier.arm_conflict(x, y, crit)
    classifier.arm_objectives(x, y, crit)
    descs = classifier.all_metric_descriptions()
    for key in classifier.training_metrics():
        if key.startswith("arm"):
            assert key in descs, f"{key} has no card"


def test_equalization_preserves_the_step_magnitude():
    """It changes the update's DIRECTION, not the effective learning rate."""
    classifier = build("prismatic9").train()
    x, y = _batch()
    crit = _halo_objectives()

    classifier.equalize_rows = False
    plain = torch.autograd.grad(
        classifier.arm_objectives(x, y, crit)["arm_surgery"], x, retain_graph=True
    )[0]
    classifier.equalize_rows = True
    equal = torch.autograd.grad(
        classifier.arm_objectives(x, y, crit)["arm_surgery"], x, retain_graph=True
    )[0]

    assert float(equal.norm()) == pytest.approx(float(plain.norm()), rel=0.05)
    # Same size, genuinely different direction.
    cos = float(equal.flatten() @ plain.flatten() / (equal.norm() * plain.norm()))
    assert cos < 0.99


def test_no_grad_forward_while_training_is_a_no_op():
    """Lazy-module init calls `model.train()` then runs a `torch.no_grad()`
    dummy pass (praxis/utils/system.py). Building an arm loss there gives a
    tensor with no grad_fn, and autograd.grad on it raises "element 0 of
    tensors does not require grad"."""
    classifier = build("prismatic9").train()
    x, y = _batch()
    crit = _halo_objectives()
    with torch.no_grad():
        assert classifier.arm_objectives(x, y, crit) == {}
        assert classifier.arm_conflict(x, y, crit) == {}
    # And grad-enabled behaviour is untouched.
    assert classifier.arm_objectives(x, y, crit)
    assert classifier.arm_conflict(x, y, crit)


def test_undetached_is_scoped_and_restores_the_surgery():
    """Every arm is detached in the blend - otherwise each gets BOTH its solo
    gradient and the mixture's responsibility-weighted one, and starvation
    comes back through the second path. The context manager must therefore
    never leave the classifier undetached, even when its body raises."""
    classifier = build("prismatic9").train()
    assert all(b.detach_in_blend for b in classifier.branches)
    assert classifier.detach_gate_input is True

    with classifier.undetached():
        assert not any(b.detach_in_blend for b in classifier.branches)
        assert classifier.detach_gate_input is False

    assert all(b.detach_in_blend for b in classifier.branches)
    assert classifier.detach_gate_input is True

    with pytest.raises(RuntimeError):
        with classifier.undetached():
            raise RuntimeError("boom")
    assert all(b.detach_in_blend for b in classifier.branches)
    assert classifier.detach_gate_input is True
