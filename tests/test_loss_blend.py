"""How the several loss terms fold into one scalar, and what the fold can see.

The fold is the last place a run's objectives can be got wrong, and it is the
one place nothing was checking. These pin three things: the anchor resolves to
a term that actually reaches the trunk, the cap only ever shrinks, and a
value-space rule extinguishes a value-neutral surrogate (the concrete reason
gradient space is the right space here).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from praxis.containers import LossContainer
from praxis.heads import HEAD_REGISTRY
from praxis.losses.conflict import ObjectiveConflict
from praxis.losses.trunk_grads import resolve_anchor, trunk_gradients
from praxis.strategies import STRATEGIES_REGISTRY
from praxis.strategies.anchor_capped import AnchorCapped


def _cfg(**over):
    base = dict(
        hidden_size=16,
        vocab_size=32,
        max_position_embeddings=64,
        encoder_type="",
        loss_func="halo",
        crystal_n=None,
        crystal_label_smoothing=None,
        tie_word_embeddings=False,
        embed_size=16,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ── the anchor ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["prismatic7", "prismatic8"])
def test_mixture_ce_reaches_the_trunk_on_an_ordinary_parallel_head(name):
    torch.manual_seed(0)
    head = HEAD_REGISTRY[name](_cfg())
    head.train()
    h = torch.randn(2, 8, 16, requires_grad=True)
    ce = F.cross_entropy(head(h).reshape(-1, 32), torch.randint(0, 32, (16,)))
    assert torch.autograd.grad(ce, h, allow_unused=True)[0] is not None


def test_surgical_head_detaches_main_from_the_trunk_entirely():
    """prismatic9 detaches every arm in the blend AND the gate's input, so the
    mixture CE trains only the gate. Anchoring the Jacobian on ``main`` there
    measures nothing at all - which is why every conflict_* series stayed dark
    through abstractinator-u."""
    torch.manual_seed(0)
    head = HEAD_REGISTRY["prismatic9"](_cfg())
    head.train()
    h = torch.randn(2, 8, 16, requires_grad=True)
    ce = F.cross_entropy(head(h).reshape(-1, 32), torch.randint(0, 32, (16,)))
    assert torch.autograd.grad(ce, h, allow_unused=True)[0] is None


def test_anchor_falls_back_to_the_surgical_row_when_main_cannot_reach():
    h = torch.randn(4, requires_grad=True)
    rows = {"arm_surgery": (h * 3).sum(), "harmonic_kl": (h * 0.5).sum()}
    live = trunk_gradients(rows, h)
    assert resolve_anchor(live) == "arm_surgery"


def test_anchor_prefers_main_whenever_main_is_live():
    h = torch.randn(4, requires_grad=True)
    rows = {"main": (h * 2).sum(), "arm_surgery": (h * 3).sum()}
    assert resolve_anchor(trunk_gradients(rows, h)) == "main"


def test_a_term_with_no_path_to_the_trunk_gets_no_row():
    """Parameter-only terms do not compete for the shared representation, so
    absence is the answer rather than a zero."""
    h = torch.randn(4, requires_grad=True)
    p = torch.randn(4, requires_grad=True)
    live = trunk_gradients({"main": (h * 2).sum(), "centers_rms": (p**2).sum()}, h)
    assert set(live) == {"main"}


def test_conflict_reports_once_the_anchor_resolves():
    """The measurement that was returning empty every step."""
    conflict = ObjectiveConflict(interval=1)
    h = torch.randn(8, requires_grad=True)
    anchor = (h * 2).sum()
    metrics = conflict.measure(
        {"arm_surgery": anchor, "harmonic_kl": -(h * 2).sum()}, h
    )
    assert metrics["conflict_harmonic_kl"] == pytest.approx(-1.0, abs=1e-5)
    assert metrics["conflict_mag_harmonic_kl"] == pytest.approx(1.0, abs=1e-5)
    assert conflict.anchor == "arm_surgery"


# ── the cap ─────────────────────────────────────────────────────────────


def _blend(rows, interval=1):
    s = AnchorCapped(interval=interval)
    s.train()
    return s


def test_cap_is_the_plain_sum_when_nothing_out_pulls_the_anchor():
    torch.manual_seed(0)
    h = torch.randn(8, requires_grad=True)
    names = ["main", "harmonic_kl"]
    losses = [(h * 4).sum(), (h * 0.1).sum()]
    s = _blend(names)
    folded = s(losses, names=names, trunk=h)
    assert float(folded.detach()) == pytest.approx(float(sum(losses).detach()), rel=1e-6)
    assert all(w == pytest.approx(1.0) for w in s._weights.values())


def test_cap_shrinks_only_the_term_that_out_pulls_the_anchor():
    h = torch.randn(8, requires_grad=True)
    names = ["main", "loud"]
    # 'loud' pulls 10x harder than the anchor at the trunk.
    losses = [(h * 1.0).sum(), (h * 10.0).sum()]
    s = AnchorCapped(interval=1)
    s.train()
    for _ in range(200):  # let the EMA settle
        s(
            [(h * 1.0).sum(), (h * 10.0).sum()],
            names=names,
            trunk=h,
        )
    assert s._weights["main"] == pytest.approx(1.0, abs=1e-3)
    assert s._weights["loud"] == pytest.approx(0.1, abs=1e-3)


def test_cap_never_amplifies():
    h = torch.randn(8, requires_grad=True)
    names = ["main", "quiet"]
    s = AnchorCapped(interval=1)
    s.train()
    for _ in range(50):
        s([(h * 5.0).sum(), (h * 0.01).sum()], names=names, trunk=h)
    assert all(w <= 1.0 + 1e-9 for w in s._weights.values())
    assert s._weights["quiet"] == pytest.approx(1.0)


def test_cap_weights_carry_no_gradient():
    """A learned weight is a route for the model to switch off an objective it
    finds inconvenient. These are constants; there is nothing to learn."""
    s = AnchorCapped(interval=1)
    s.train()
    assert list(s.parameters()) == []


def test_cap_falls_back_to_the_plain_sum_without_names():
    """The layer-wise fold calls strategies positionally and has no trunk."""
    losses = [torch.tensor(1.0), torch.tensor(2.0)]
    assert float(AnchorCapped()(losses)) == pytest.approx(3.0)


def test_cap_is_registered_and_reports_its_weights():
    s = STRATEGIES_REGISTRY["capped"]()
    s.train()
    h = torch.randn(8, requires_grad=True)
    s([(h * 1.0).sum(), (h * 4.0).sum()], names=["main", "loud"], trunk=h)
    m = s.training_metrics()
    assert "blend_w_main" in m and "blend_w_loud" in m and "blend_w_min" in m


# ── why gradient space, not value space ─────────────────────────────────


def test_value_space_weighting_explodes_on_a_value_neutral_surrogate():
    """``arm_surgery`` is ``g - g.detach()``: value identically 0, gradient the
    only task signal the trunk gets under prismatic9.

    Kendall's ``exp(-s) L + s`` has gradient ``1 - exp(-s) L`` in ``s``, so
    ``s`` settles at ``log L`` and the weight at ``1 / L``. At ``L = 0`` the
    gradient is ``+1`` forever: ``s`` runs to minus infinity and the weight
    grows WITHOUT BOUND. The same thing happens more slowly to any auxiliary
    that converges - a KL or an isotropy term approaching zero is rewarded
    with an ever-larger share of the update. That is the divergence, not a
    mute.

    The cap reads the gradient instead, so a zero value costs it nothing.
    """
    s = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([s], lr=0.1)
    zero_valued = torch.tensor(0.0)
    weights = []
    for _ in range(200):
        opt.zero_grad()
        (torch.exp(-s) * zero_valued + s).sum().backward()
        opt.step()
        weights.append(float(torch.exp(-s).detach()))
    assert weights[-1] > 1e6  # unbounded growth, no fixed point
    assert weights[-1] > weights[0]

    h = torch.randn(8, requires_grad=True)
    g = (h * 3.0).sum()
    surrogate = g - g.detach()
    assert float(surrogate.detach()) == 0.0
    blend = AnchorCapped(interval=1)
    blend.train()
    blend([surrogate, (h * 1.0).sum()], names=["arm_surgery", "harmonic_kl"], trunk=h)
    assert blend._weights["arm_surgery"] == pytest.approx(1.0)


def test_a_converging_auxiliary_takes_over_a_value_space_blend():
    """The general form of the same defect: as an auxiliary gets easier its
    weight rises as 1/L, so the term with the least left to say ends up
    shouting. The cap cannot do this - it only ever shrinks."""
    s = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([s], lr=0.05)
    weights = []
    for step in range(400):
        opt.zero_grad()
        converging = torch.tensor(max(1e-3, 1.0 * (0.99**step)))
        (torch.exp(-s) * converging + s).sum().backward()
        opt.step()
        weights.append(float(torch.exp(-s).detach()))
    # Starts at 1 and climbs the whole way: the fixed point is 1/L, and L is
    # still falling, so there is nothing holding the weight down.
    assert weights[0] == pytest.approx(1.0, abs=0.1)
    assert weights[-1] > 20.0
    assert weights[-1] > weights[len(weights) // 2] > weights[0]


def test_container_names_and_values_stay_aligned():
    c = LossContainer()
    c.add_loss("main", torch.tensor(1.0))
    c.add_loss("mtp", torch.tensor(2.0))
    names, values = c.get_named_losses()
    assert names == ["main", "mtp"]
    assert [float(v) for v in values] == [1.0, 2.0]
    assert values == c.get_loss_values()


def test_first_reading_is_taken_not_approached():
    """Approaching a measured cap from 1.0 would spend the run's first
    thousand steps applying a cap nothing measured."""
    h = torch.randn(8, requires_grad=True)
    s = AnchorCapped(interval=1)
    s.train()
    s([(h * 1.0).sum(), (h * 4.0).sum()], names=["main", "loud"], trunk=h)
    assert s._weights["loud"] == pytest.approx(0.25, abs=1e-3)


def test_caps_survive_a_checkpoint_round_trip():
    h = torch.randn(8, requires_grad=True)
    a = AnchorCapped(interval=1)
    a.train()
    a([(h * 1.0).sum(), (h * 4.0).sum()], names=["main", "loud"], trunk=h)

    b = AnchorCapped(interval=1)
    b.load_state_dict(a.state_dict())
    assert b._weights == a._weights
    assert b.anchor == a.anchor == "main"
