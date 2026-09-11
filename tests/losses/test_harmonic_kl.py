"""HarmonicKLRegularizer: KL between the output readout and a slow EMA of itself."""

import copy

import pytest
import torch
import torch.nn as nn

from praxis.losses.harmonic_kl import KL_WEIGHT, HarmonicKLRegularizer


def _readout(vocab=17, dim=8, bias=True):
    return nn.Linear(dim, vocab, bias=bias)


class _CenterReadout(nn.Module):
    """Stand-in for CrystalGeometry: a parameterised readout with NO `.weight`.

    This is the shape that made the first version of this regularizer a silent
    no-op on every abstractinator config (classifier_type: prismatic4 ->
    ParallelClassifier -> CrystalGeometry, whose only parameter is `centers`).
    Duck-typing `.weight` found nothing and returned zero forever.
    """

    def __init__(self, vocab=17, dim=8):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(vocab, dim))

    def forward(self, x):
        d = torch.cdist(x.reshape(-1, x.shape[-1]).float(), self.centers.float())
        return (-(d**2)).view(*x.shape[:-1], self.centers.size(0))


def _hidden(b=2, t=6, dim=8):
    return torch.randn(b, t, dim, requires_grad=True)


def test_first_forward_is_zero_and_seeds_the_teacher():
    reg = HarmonicKLRegularizer()
    clf = _readout()
    loss = reg(_hidden(), None, scorer=clf)
    assert float(loss) == 0.0
    assert reg._ema_ready
    assert torch.equal(reg._ema_state().get("weight"), clf.weight.detach())
    assert torch.equal(reg._ema_state().get("bias"), clf.bias.detach())


def test_works_on_a_readout_without_a_weight_attribute():
    """The regression that mattered: CrystalGeometry-shaped readouts.

    An earlier version duck-typed `.weight` and silently returned zero for the
    entire abstractinator line. The EMA is now generic over named_parameters."""
    reg = HarmonicKLRegularizer()
    clf = _CenterReadout()
    assert not hasattr(clf, "weight")

    assert float(reg(_hidden(), None, scorer=clf)) == 0.0
    assert reg._ema_ready, "must seed from `centers`, not give up"
    assert torch.equal(reg._ema_state().get("centers"), clf.centers.detach())

    with torch.no_grad():
        clf.centers.add_(torch.randn_like(clf.centers))
    h = _hidden()
    loss = reg(h, None, scorer=clf)
    assert loss.item() > 0
    loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert clf.centers.grad is not None and torch.isfinite(clf.centers.grad).all()


def test_zero_drift_when_the_readout_has_not_moved():
    reg = HarmonicKLRegularizer()
    clf = _readout()
    reg(_hidden(), None, scorer=clf)  # seed
    loss = reg(_hidden(), None, scorer=clf)
    assert float(loss) == 0.0
    assert reg.training_metrics()["harmonic_drift"] == 0.0


def test_drift_is_positive_after_the_readout_moves():
    reg = HarmonicKLRegularizer()
    clf = _readout()
    reg(_hidden(), None, scorer=clf)  # seed
    with torch.no_grad():
        clf.weight.add_(torch.randn_like(clf.weight))
    h = _hidden()
    loss = reg(h, None, scorer=clf)
    m = reg.training_metrics()
    assert m["harmonic_drift"] > 0
    assert loss.item() == pytest.approx(KL_WEIGHT * m["harmonic_drift"], rel=1e-5)
    assert set(m) == {"harmonic_kl_loss", "harmonic_drift", "harmonic_live_entropy"}
    loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert clf.weight.grad is not None and torch.isfinite(clf.weight.grad).all()


def test_teacher_advances_only_after_scoring():
    """A step must never be measured against a teacher that already absorbed it."""
    reg = HarmonicKLRegularizer(decay=0.5)
    clf = _readout()
    reg(_hidden(), None, scorer=clf)  # seed
    before = reg._ema_state()["weight"].clone()
    with torch.no_grad():
        clf.weight.add_(1.0)
    reg(_hidden(), None, scorer=clf)
    # EMA moved toward the new weights, but is not equal to them.
    now = reg._ema_state()["weight"]
    assert not torch.equal(now, before)
    assert not torch.equal(now, clf.weight.detach())


def test_noop_paths_return_zero():
    h = _hidden()
    # No scorer supplied (the other regularizers' call shape).
    assert float(HarmonicKLRegularizer()(h, None)) == 0.0
    # 2-D input.
    assert (
        float(HarmonicKLRegularizer()(torch.randn(4, 8), None, scorer=_readout()))
        == 0.0
    )
    # Parameter-free readout: disables itself rather than pretending to work.
    reg = HarmonicKLRegularizer()
    assert float(reg(h, None, scorer=nn.Identity())) == 0.0
    assert reg._disabled


def test_width_mismatch_disables_instead_of_crashing_training():
    """A readout that cannot consume the reps must not take the run down."""
    reg = HarmonicKLRegularizer()
    bad = _readout(dim=99)
    assert float(reg(_hidden(), None, scorer=bad)) == 0.0  # seeds
    assert float(reg(_hidden(), None, scorer=bad)) == 0.0  # raises -> disabled
    assert reg._disabled


def test_reseeds_when_the_readout_shape_changes():
    reg = HarmonicKLRegularizer()
    reg(_hidden(), None, scorer=_readout(vocab=17))
    wide = _readout(vocab=33)
    loss = reg(_hidden(), None, scorer=wide)
    assert float(loss) == 0.0  # re-seeded, so nothing to compare against yet
    assert reg._ema_state()["weight"].shape == wide.weight.shape


def test_buffers_are_non_persistent():
    """Resume must re-seed the teacher from the live readout - a zero penalty,
    not a stale one. (praxis/policies/engagement.py's baseline was the opposite
    mistake: a non-checkpointed value that cold-started far from the truth.)"""
    reg = HarmonicKLRegularizer()
    reg(_hidden(), None, scorer=_readout())
    assert reg._ema_state(), "teacher must exist"
    assert reg.state_dict() == {}


class _RowCounter(nn.Linear):
    """A Linear readout that records how many positions each call scores."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []

    def forward(self, x):
        self.rows.append(x.shape[0])
        return super().forward(x)


def test_position_subsampling_caps_cost(monkeypatch):
    monkeypatch.setattr("praxis.losses.harmonic_kl.MAX_POSITIONS", 8)
    reg = HarmonicKLRegularizer()
    clf = _RowCounter(8, 17)
    h = _hidden(b=2, t=6)  # 12 positions > the cap
    reg(h, None, scorer=clf)  # seed
    with torch.no_grad():
        clf.weight.add_(torch.randn_like(clf.weight))
    loss = reg(h, None, scorer=clf)
    assert loss.item() > 0
    # Live and teacher readouts both scored the capped subsample.
    assert clf.rows == [8, 8]


def test_pad_positions_are_dropped_when_ids_align():
    """Padded positions change nothing; a live one does. Each reading is
    scored against its own copy of the same seeded teacher."""
    reg = HarmonicKLRegularizer(pad_id=0)
    clf = _readout()
    h = torch.randn(2, 6, 8)
    ids = torch.ones(2, 6, dtype=torch.long)
    ids[:, 3:] = 0  # padded tail
    reg(h, ids, scorer=clf)  # seed
    with torch.no_grad():
        clf.weight.add_(torch.randn_like(clf.weight))

    def score(states):
        return copy.deepcopy(reg)(states, ids, scorer=clf).item()

    base = score(h)
    padded = h.clone()
    padded[:, 3:] += 5.0 * torch.randn(2, 3, 8)
    live = h.clone()
    live[:, 0] += 5.0 * torch.randn(2, 8)
    assert base > 0
    assert score(padded) == pytest.approx(base, rel=1e-6)
    assert score(live) != pytest.approx(base, rel=1e-3)
    # All-pad is a no-op rather than an empty reduction.
    assert torch.isfinite(reg(h, torch.zeros(2, 6, dtype=torch.long), scorer=clf))


def test_bias_free_readout_is_supported():
    reg = HarmonicKLRegularizer()
    clf = _readout(bias=False)
    reg(_hidden(), None, scorer=clf)  # seed
    assert set(reg._ema_state()) == {"weight"}
    with torch.no_grad():
        clf.weight.add_(torch.randn_like(clf.weight))
    assert float(reg(_hidden(), None, scorer=clf)) > 0


def test_harmonic_kl_names_the_readout_it_actually_watches(tiny_model, train_step):
    """The misreading this exists to stop: on a multi-arm classifier the target
    is HALO's centroids, not the harmonic field."""
    m = tiny_model(regularizers=["harmonic_kl"])
    assert repr(m.criterion.harmonic_kl) == "HarmonicKLRegularizer(target=readout)"
    train_step(m)
    target = m.criterion.harmonic_kl._target
    assert target.startswith("HaloGeometry(")
    assert "centers" in target and "gamma" in target
