"""prismatic9: the arms train on their own objectives, PCGrad on the trunk.

The mixture of softmaxes is a standard and correct way to COMBINE predictions,
and prismatic9 does not change it. What it changes is that the mixture also
decided, as a side effect, how much each arm got TRAINED - cross-entropy
reaches arm i scaled by its posterior responsibility, so an arm the gate stops
trusting stops receiving gradient. abstractinator-n drove one to 6.3e-08.

Here each arm gets its own cross-entropy and the trunk receives one
PCGrad-combined gradient over those objectives instead of their plain sum.
"""

import pytest
import torch
import torch.nn as nn

from praxis.heads import HEAD_REGISTRY
from praxis.heads.parallel import (
    ARM_CONFLICT_INTERVAL,
    ParallelHead,
    SurgicalParallelHead,
    _pcgrad,
)
from praxis.losses import Objectives
from praxis.losses.cross_entropy import CrossEntropyLoss
from praxis.losses.halo import HALOLoss
from tests.test_prismatic8 import Cfg, Enc


def objectives(main=None):
    """The loss container a head is handed, as modeling.py builds it.

    Every arm's row comes out of here: ``arm_ce`` for the plain arms, the
    main criterion's geometry for the HALO one. Omitting ``main`` is the
    no-criterion case - the HALO arm has nothing to score with.
    """
    terms = Objectives()
    if main is not None:
        terms.register("main", main)
    terms.register("arm_ce", CrossEntropyLoss())
    return terms


def build(name="prismatic9"):
    torch.manual_seed(0)
    return HEAD_REGISTRY[name](Cfg(), encoder=Enc())


def batch(b=2, t=5, d=48, v=32):
    torch.manual_seed(1)
    return torch.randn(b, t, d, requires_grad=True), torch.randint(0, v, (b, t))


# ── PCGrad itself ──────────────────────────────────────────────────────────


def test_pcgrad_is_the_plain_sum_when_nothing_conflicts():
    """Its designed no-op. This is what makes it safe as a default rather than
    a bet: it can only act where there is something to act on."""
    g = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0])]
    assert torch.allclose(_pcgrad(g), sum(g))


def test_pcgrad_removes_the_conflicting_component():
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0])
    out = _pcgrad([a, b])
    # a loses its projection onto b, b loses its projection onto a.
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
    training path, which this codebase has been bitten by before."""
    torch.manual_seed(3)
    g = [torch.randn(8) for _ in range(4)]
    assert torch.allclose(_pcgrad(g), _pcgrad(list(reversed(g))), atol=1e-6)


def test_pcgrad_matches_the_paper_exactly_for_two_rows():
    """Where the deviation cannot bite: with two rows there is no order to
    depend on, so this is Yu et al. unmodified."""
    a, b = torch.tensor([1.0, 0.0, 2.0]), torch.tensor([-1.0, 1.0, 0.5])
    a_p = a - (a @ b) / (b @ b) * b
    b_p = b - (b @ a) / (a @ a) * a
    assert torch.allclose(_pcgrad([a, b]), a_p + b_p, atol=1e-6)


def test_pcgrad_preserves_shape():
    g = [torch.randn(2, 5, 7) for _ in range(3)]
    assert _pcgrad(g).shape == (2, 5, 7)


# ── The measurement, which every ParallelHead gets ─────────────────────────


def test_arm_conflict_reports_a_full_jacobian_on_a_plain_parallel_head():
    """Measurement is universal: prismatic8 reports its arm Jacobian without
    any change to how it trains."""
    head = build("prismatic8").train()
    x, y = batch()
    m = head.arm_conflict(x, y, objectives(HALOLoss(vocab_size=32)))
    # ALL THREE arms are rows. The HALO arm's row is its own geometric
    # objective, not an invented CE - a Jacobian row is any objective's
    # gradient w.r.t. the shared representation and need not be a CE.
    assert set(k for k in m if k.startswith("arm_cos_") and k != "arm_cos_min") == {
        "arm_cos_01",
        "arm_cos_02",
        "arm_cos_12",
    }
    assert m["arm_cos_min"] == min(
        m[k] for k in m if k.startswith("arm_cos_") and k != "arm_cos_min"
    )
    for i in range(3):
        assert 0.0 <= m[f"arm_grad_share_{i}"] <= 1.0
    assert max(m[f"arm_grad_share_{i}"] for i in range(3)) == pytest.approx(1.0)


def test_arm_conflict_is_sampled_and_holds_between_samples():
    head = build("prismatic8").train()
    x, y = batch()
    terms = objectives()
    first = dict(head.arm_conflict(x, y, terms))
    assert first
    for _ in range(ARM_CONFLICT_INTERVAL - 1):
        assert head.arm_conflict(x, y, terms) == first
    assert head._arm_step == ARM_CONFLICT_INTERVAL


def test_arm_conflict_is_inert_at_eval_and_without_labels():
    head = build("prismatic8")
    x, y = batch()
    assert head.eval().arm_conflict(x, y, objectives()) == {}
    assert head.train().arm_conflict(x, None, objectives()) == {}


def test_arm_conflict_does_not_disturb_the_training_graph():
    """The diagnostic runs on a detached leaf, so the real backward that
    follows must be unaffected."""
    head = build("prismatic8").train()
    x, y = batch()
    head.arm_conflict(x, y, objectives())
    head(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ── The intervention, which only prismatic9 gets ───────────────────────────


def test_prismatic8_emits_no_arm_objectives():
    head = build("prismatic8").train()
    x, y = batch()
    assert head.arm_objectives(x, y, objectives(HALOLoss(vocab_size=32))) == {}
    assert head.arm_surgery is False


def test_prismatic9_emits_one_ce_per_arm_plus_the_surrogate():
    head = build().train()
    x, y = batch()
    out = head.arm_objectives(x, y, objectives(HALOLoss(vocab_size=32)))
    # Every arm is a row, HALO included, so nothing reaches the trunk outside
    # the arbitration.
    assert set(out) == {"arm0_loss", "arm1_loss", "arm2_loss", "arm_surgery"}
    assert isinstance(head, SurgicalParallelHead)


def test_surgery_refuses_rather_than_shipping_a_partial_jacobian():
    """An objective that reaches the shared representation but sits OUTSIDE the
    arbitration routes around the very thing the surgery does. With no
    criterion the HALO arm cannot supply its row, so the whole intervention
    stands down instead of arbitrating two rows out of three."""
    head = build().train()
    x, y = batch()
    assert head.arm_objectives(x, y, objectives()) == {}
    assert head._arm_gap is True
    assert head.arm_objectives(x, y, objectives(HALOLoss(vocab_size=32)))


def test_the_gate_cannot_bypass_the_surgery():
    """The gate's CE would otherwise be a fourth uncorrected path into the
    trunk - the same defect as an excluded arm, just quieter."""
    nine, eight = build("prismatic9").train(), build("prismatic8").train()
    assert nine.detach_gate_input is True
    assert eight.detach_gate_input is False
    x, _ = batch()
    # Under prismatic9 the blended logits carry no trunk gradient at all: every
    # arm is detached in the blend and the gate reads a detached trunk.
    g = torch.autograd.grad(nine(x).sum(), x, allow_unused=True)[0]
    assert g is None or g.abs().sum() == 0
    # prismatic8 keeps the ordinary path.
    g8 = torch.autograd.grad(eight(x).sum(), x, allow_unused=True)[0]
    assert g8 is not None and g8.abs().sum() > 0


def test_halo_row_is_its_own_objective_not_a_cross_entropy():
    head = build().train()
    x, y = batch()
    crit = objectives(HALOLoss(vocab_size=32))
    out = head.arm_objectives(x, y, crit)
    halo_arm = head.branches[2]
    direct = halo_arm.arm_loss(x.detach().requires_grad_(True), y, crit)
    assert torch.allclose(out["arm2_loss"], direct, atol=1e-5)
    # And it is NOT what a CE on the arm would give.
    ce = torch.nn.functional.cross_entropy(
        (
            halo_arm(x)[..., :-1, :].reshape(-1, 32).float()
            if halo_arm(x).shape[-2] != y.shape[-1]
            else halo_arm(x).reshape(-1, 32).float()
        ),
        y.reshape(-1),
    )
    assert not torch.allclose(direct, ce, atol=1e-3)


def test_the_surrogate_contributes_no_value_to_the_loss():
    """Its magnitude is arbitrary - it exists only to carry a gradient - so it
    must not land in the loss curve people read."""
    head = build().train()
    x, y = batch()
    out = head.arm_objectives(x, y, objectives(HALOLoss(vocab_size=32)))
    assert float(out["arm_surgery"].detach()) == pytest.approx(0.0, abs=1e-9)
    (g,) = torch.autograd.grad(out["arm_surgery"], x, retain_graph=True)
    assert g.abs().sum() > 0, "value-neutral must not mean gradient-neutral"


def test_the_surrogate_delivers_exactly_the_pcgrad_gradient_to_the_trunk():
    """``d/dz (z * g_hat).sum() == g_hat``. This is the whole mechanism: the
    trunk receives the combined gradient without a second backward through it."""
    head = build().train()
    head.equalize_rows = False  # the identity is over the raw PCGrad result
    x, y = batch()
    out = head.arm_objectives(x, y, objectives(HALOLoss(vocab_size=32)))
    (g,) = torch.autograd.grad(out["arm_surgery"], x, retain_graph=True)

    # Recompute the expected combination independently.
    z = head._stem_out(x)
    z_d = z.detach().requires_grad_(True)
    trunk_d = x.detach().requires_grad_(True)
    grads = []
    crit = objectives(HALOLoss(vocab_size=32))
    for b in head.branches:
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
    each backward crosses one small classifier, never the trunk."""
    head = build().train()
    x, y = batch()
    out = head.arm_objectives(x, y, objectives(HALOLoss(vocab_size=32)))
    (g,) = torch.autograd.grad(
        out["arm0_loss"], x, allow_unused=True, retain_graph=True
    )
    assert g is None, "an arm's solo CE must not reach the trunk"

    crystal = head.branches[0]
    params = [p for p in crystal.parameters() if p.requires_grad]
    gs = torch.autograd.grad(
        out["arm0_loss"], params, allow_unused=True, retain_graph=True
    )
    assert any(t is not None and t.abs().sum() > 0 for t in gs)


def test_every_arm_is_detached_in_the_blend():
    """Otherwise each arm gets BOTH its solo gradient and the mixture's
    responsibility-weighted one, and the starvation comes back through the
    second path."""
    head = build()
    assert all(b.detach_in_blend for b in head.branches)


def test_a_starved_arm_still_gets_full_gradient():
    """The point of the design: an arm's training signal does not pass through
    the gate at all, so gate collapse cannot starve it. abstractinator-n drove
    arm 1 to a gate share of 6.3e-08, which under the mixture is an arm that
    can never recover."""
    head = build().train()
    x, y = batch()
    crit = objectives(HALOLoss(vocab_size=32))
    before = head.arm_objectives(x, y, crit)
    with torch.no_grad():
        head.gate.weight.mul_(0).add_(torch.randn_like(head.gate.weight) * 100)
    after = head.arm_objectives(x, y, crit)
    # Wrecking the gate leaves every arm's own objective bit-identical.
    for k in ("arm0_loss", "arm1_loss", "arm2_loss"):
        assert torch.equal(before[k], after[k]), k

    starved = [p for p in head.branches[1].parameters() if p.requires_grad]
    gs = torch.autograd.grad(after["arm1_loss"], starved, allow_unused=True)
    assert any(t is not None and t.abs().sum() > 0 for t in gs)


def test_forward_and_inference_are_unchanged_vs_prismatic8():
    """Identical construction, identical predictions: prismatic9 changes
    training only. detach_in_blend is a no-op outside training."""
    nine, eight = build("prismatic9").eval(), build("prismatic8").eval()
    x, _ = batch()
    with torch.no_grad():
        assert torch.allclose(nine(x), eight(x), atol=1e-6)


def test_repr_names_the_intervention():
    assert build().compose_repr().startswith("SurgicalParallel")
    assert not build("prismatic8").compose_repr().startswith("Surgical")


def test_metrics_and_descriptions_line_up():
    head = build().train()
    x, y = batch()
    crit = objectives(HALOLoss(vocab_size=32))
    head.arm_conflict(x, y, crit)
    head.arm_objectives(x, y, crit)
    metrics = head.training_metrics()
    descs = head.all_metric_descriptions()
    for key in metrics:
        if key.startswith("arm"):
            assert key in descs, f"{key} has no card"


# ── Override: the measurement that decides whether magnitude actually bit ──


def test_override_is_zero_when_arms_agree():
    """Rows pointing the same way cannot overrule each other whatever their
    relative sizes, which is the whole point: magnitude alone is not harm."""
    from praxis.heads.parallel import ParallelHead as P

    big = torch.tensor([20.0, 20.0, 20.0])
    small = torch.tensor([1.0, 1.0, 1.0])
    out = P._override_shares(
        [small, big], [float(small.norm()), float(big.norm())], [0, 1]
    )
    assert out["arm_override_0"] == pytest.approx(0.0)
    assert out["arm_override_1"] == pytest.approx(0.0)


def test_override_catches_a_loud_row_reversing_a_quiet_one():
    """A 20:1 magnitude ratio with opposed signs: the plain sum points the
    wrong way for the quiet arm on every coordinate."""
    from praxis.heads.parallel import ParallelHead as P

    quiet = torch.tensor([1.0, 1.0, 1.0])
    loud = torch.tensor([-20.0, -20.0, -20.0])
    out = P._override_shares(
        [quiet, loud], [float(quiet.norm()), float(loud.norm())], [0, 1]
    )
    assert out["arm_override_0"] == pytest.approx(1.0)
    assert out["arm_override_1"] == pytest.approx(0.0)
    # And PCGrad is what recovers it: fully opposed rows cancel to zero rather
    # than letting the loud one dictate.
    assert out["arm_override_pcg_0"] == pytest.approx(0.0)


def test_override_is_mass_weighted_not_counted():
    """A flipped coordinate the arm barely cared about is not an override."""
    from praxis.heads.parallel import ParallelHead as P

    # Arm 0 cares almost entirely about coord 0, which survives; coord 1 flips.
    a = torch.tensor([100.0, 0.01])
    b = torch.tensor([1.0, -50.0])
    out = P._override_shares([a, b], [float(a.norm()), float(b.norm())], [0, 1])
    assert out["arm_override_0"] < 0.001, "a counted metric would report 50%"


def test_override_ships_on_a_plain_parallel_head_too():
    """Measurement is universal, so -n-style runs can be read the same way."""
    head = build("prismatic8").train()
    x, y = batch()
    m = head.arm_conflict(x, y, objectives(HALOLoss(vocab_size=32)))
    for i in range(3):
        assert 0.0 <= m[f"arm_override_{i}"] <= 1.0
        assert 0.0 <= m[f"arm_override_pcg_{i}"] <= 1.0


def test_equalizing_rows_gives_a_drowned_arm_a_vote_back():
    """The measured problem was not conflict, it was drowning: a 20-65x row
    that is merely ORTHOGONAL to its neighbours leaves PCGrad nothing to
    project, and 0.5 override is the coin-flip null for a small row against a
    dominant one - it means no vote, not opposition."""
    from praxis.heads.parallel import ParallelHead as P

    torch.manual_seed(7)
    quiet = torch.randn(4096) * 0.05
    other = torch.randn(4096) * 0.05
    loud = torch.randn(4096) * 1.0  # ~20x, and orthogonal by construction
    rows = [quiet, other, loud]
    norms = [float(g.norm()) for g in rows]
    m = P._override_shares(rows, norms, [0, 1, 2])

    # Drowned: the quiet rows sit at the coin-flip null under the plain sum...
    assert 0.40 < m["arm_override_0"] < 0.60
    # ...and PCGrad cannot help, because there is no conflict to project.
    assert abs(m["arm_override_pcg_0"] - m["arm_override_0"]) < 0.05
    # Equalizing rows is what restores their say.
    assert m["arm_override_eq_0"] < 0.30
    assert m["arm_override_eq_1"] < 0.30
    # And the loud row pays for it: it stops being the only voice.
    assert m["arm_override_eq_2"] > m["arm_override_2"]


def test_equalization_preserves_the_step_magnitude():
    """It changes the update's DIRECTION, not the effective learning rate."""
    head = build().train()
    x, y = batch()
    crit = objectives(HALOLoss(vocab_size=32))

    head.equalize_rows = False
    plain = torch.autograd.grad(
        head.arm_objectives(x, y, crit)["arm_surgery"], x, retain_graph=True
    )[0]
    head.equalize_rows = True
    equal = torch.autograd.grad(
        head.arm_objectives(x, y, crit)["arm_surgery"], x, retain_graph=True
    )[0]

    assert float(equal.norm()) == pytest.approx(float(plain.norm()), rel=0.05)
    # Same size, genuinely different direction.
    cos = float(equal.flatten() @ plain.flatten() / (equal.norm() * plain.norm()))
    assert cos < 0.99


def test_equalization_is_on_for_prismatic9_and_absent_elsewhere():
    assert build("prismatic9").equalize_rows is True
    assert not hasattr(build("prismatic8"), "equalize_rows")


def test_no_grad_forward_while_training_is_a_no_op():
    """Lazy-module init calls `model.train()` then runs a `torch.no_grad()`
    dummy pass (praxis/utils/system.py). Building an arm loss there gives a
    tensor with no grad_fn, and autograd.grad on it raises "element 0 of
    tensors does not require grad" - which is exactly how -o died at startup."""
    head = build().train()
    x, y = batch()
    crit = objectives(HALOLoss(vocab_size=32))
    with torch.no_grad():
        assert head.arm_objectives(x, y, crit) == {}
        assert head.arm_conflict(x, y, crit) == {}
    # And grad-enabled behaviour is untouched.
    assert head.arm_objectives(x, y, crit)
    assert head.arm_conflict(x, y, crit)


def test_full_model_survives_the_lazy_init_pass():
    """End-to-end reproduction of the -o startup crash: train() + no_grad."""
    from praxis import PraxisConfig
    from praxis.modeling import PraxisForCausalLM

    cfg = PraxisConfig(
        vocab_size=1000,
        hidden_size=32,
        embed_size=32,
        num_heads=4,
        depth=2,
        max_length=128,
        decoder_type="sequential",
        encoder_type=None,
        head_type="prismatic9",
        loss_func="halo",
    )
    torch.manual_seed(0)
    m = PraxisForCausalLM(cfg)
    m.train()
    ids = torch.ones((2, 16), dtype=torch.long)
    with torch.no_grad():
        out = m(input_ids=ids, labels=ids[..., 1:].contiguous())
    assert out.loss is not None
    # The real training step still works afterwards.
    out = m(input_ids=ids, labels=ids[..., 1:].contiguous())
    out.loss.backward()
    assert any(p.grad is not None for p in m.parameters())


def test_validation_loss_stays_comparable_to_prismatic8():
    """prismatic9 flips `composite_geometry` off so HALO's geometric term is
    not double-counted (the head owns it as a Jacobian row). That suppression
    must be TRAINING-ONLY: at eval nothing replaces the term, so zeroing it
    there just deletes a component of val_loss and makes the -n -> -o
    comparison meaningless. Caught in the wild - -o's first validation point
    landed several nats below its predecessor."""
    from praxis import PraxisConfig
    from praxis.modeling import PraxisForCausalLM

    def val_loss(head):
        c = PraxisConfig(
            vocab_size=1000,
            hidden_size=32,
            embed_size=32,
            num_heads=4,
            depth=2,
            max_length=128,
            decoder_type="sequential",
            encoder_type=None,
            head_type=head,
            loss_func="halo",
        )
        torch.manual_seed(0)
        m = PraxisForCausalLM(c).eval()
        ids = torch.arange(16).remainder(900).unsqueeze(0).repeat(2, 1)
        with torch.no_grad():
            out = m(input_ids=ids, labels=ids[:, 1:].contiguous())
        return float(out.loss), m

    eight, m8 = val_loss("prismatic8")
    nine, m9 = val_loss("prismatic9")
    # The criterion is configured differently...
    assert m8.criterion.main.composite_geometry is True
    assert m9.criterion.main.composite_geometry is False
    # ...but at EVAL both must score the same composite objective.
    assert nine == pytest.approx(
        eight, rel=1e-4
    ), f"val loss diverged: prismatic8 {eight}, prismatic9 {nine}"


def test_geometry_is_still_suppressed_during_training():
    """The other half: the double-count the flag exists to prevent."""
    from praxis.losses.halo import HALOLoss
    from praxis.heads.halo import HaloHead
    from tests.test_prismatic8 import Cfg, Enc

    torch.manual_seed(0)
    arm = HaloHead(Cfg(), encoder=Enc())
    crit = HALOLoss(vocab_size=32)
    x = torch.randn(2, 5, 48)
    y = torch.randint(0, 32, (2, 4))
    kw = dict(
        logits=arm(x)[..., :-1, :].contiguous(),
        labels=y,
        embeddings=x[..., :-1, :].contiguous(),
        classifier=arm.classifier,
    )
    crit.train()
    full = float(crit(**kw))
    crit.composite_geometry = False
    ce_only = float(crit(**kw))
    assert ce_only < full, "training-mode suppression stopped working"
    crit.eval()
    assert float(crit(**kw)) == pytest.approx(full, rel=1e-4)


def test_mtp_still_trains_under_a_surgical_head():
    """Detaching every arm in the blend severs the path from the head's OUTPUT
    back to its INPUT - and MTP classifies its draft states with that same head,
    so its loss reached nothing at all: not the arms, not the trunk, not even
    MTP's own bank. Caught in the wild from mtp_field_* series sitting frozen at
    their initialization with slope exactly 0."""
    from praxis import PraxisConfig
    from praxis.modeling import PraxisForCausalLM

    def run(head):
        cfg = PraxisConfig(
            vocab_size=1000,
            hidden_size=32,
            embed_size=32,
            num_heads=4,
            depth=2,
            max_length=128,
            decoder_type="sequential",
            encoder_type=None,
            head_type=head,
            loss_func="halo",
            mtp_depth=3,
            mtp_type="per_depth",
        )
        torch.manual_seed(0)
        m = PraxisForCausalLM(cfg).train()
        ids = torch.randint(0, 1000, (2, 16))
        m(input_ids=ids, labels=ids[:, 1:].contiguous()).loss.backward()
        live = [
            n
            for n, p in m.mtp.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        ]
        return m, live, sum(1 for _ in m.mtp.named_parameters())

    _, live8, total = run("prismatic8")
    m9, live9, _ = run("prismatic9")
    assert len(live8) == total, "baseline broke; the comparison is meaningless"
    assert len(live9) == total, f"MTP starved under prismatic9: {len(live9)}/{total}"

    # And the fallback path, for a head that has no undetached() at all. This
    # branch was a latent NameError (contextlib was never imported in
    # modeling.py) and no existing test reached it.
    m, live, total = run("forward")
    assert not hasattr(m.head, "undetached")
    assert len(live) == total


def test_undetached_is_scoped_and_restores_the_surgery():
    """The context manager must not leave the head permanently undetached -
    that would hand the arms back the mixture's responsibility-weighted
    gradient and undo the whole intervention."""
    head = build().train()
    assert all(b.detach_in_blend for b in head.branches)
    assert head.detach_gate_input is True

    with head.undetached():
        assert not any(b.detach_in_blend for b in head.branches)
        assert head.detach_gate_input is False

    assert all(b.detach_in_blend for b in head.branches)
    assert head.detach_gate_input is True

    # And it restores even when the body raises.
    with pytest.raises(RuntimeError):
        with head.undetached():
            raise RuntimeError("boom")
    assert all(b.detach_in_blend for b in head.branches)
    assert head.detach_gate_input is True


def test_the_main_loss_still_reaches_only_the_gate():
    """The property the detaching exists for, re-asserted after the MTP fix:
    outside `undetached()`, the blended logits carry no gradient to the trunk."""
    head = build().train()
    x, _ = batch()
    g = torch.autograd.grad(head(x).sum(), x, allow_unused=True)[0]
    assert g is None or g.abs().sum() == 0
