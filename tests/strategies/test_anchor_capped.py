"""AnchorCapped: cap every term's pull on the trunk at the anchor's, w = min(1, ||g_anchor|| / ||g_i||).

Pins that the cap only ever shrinks, reads gradients rather than loss values
(so a value-neutral surrogate is untouched), smooths across refreshes, and
survives a checkpoint.
"""

import pytest
import torch

from praxis import registry
from praxis.strategies.anchor_capped import (
    DECAY,
    AnchorCapped,
    blend_metric_descriptions,
)


def _capped():
    s = AnchorCapped(interval=1)
    s.train()
    return s


def test_cap_is_the_plain_sum_when_nothing_out_pulls_the_anchor():
    h = torch.randn(8, requires_grad=True)
    names = ["main", "harmonic_kl"]
    losses = [(h * 4).sum(), (h * 0.1).sum()]
    s = _capped()
    folded = s(losses, names=names, trunk=h)
    assert folded.item() == pytest.approx(sum(losses).item(), rel=1e-6)
    assert all(w == pytest.approx(1.0) for w in s._weights.values())


def test_cap_shrinks_only_the_term_that_out_pulls_the_anchor():
    """The first reading is taken outright (approaching it from 1.0 would spend
    a run's first thousand steps applying a cap nothing measured); later
    readings are smoothed by DECAY."""
    h = torch.randn(8, requires_grad=True)
    names = ["main", "loud"]
    s = _capped()

    s([(h * 1.0).sum(), (h * 4.0).sum()], names=names, trunk=h)
    assert s._weights["main"] == pytest.approx(1.0)
    assert s._weights["loud"] == pytest.approx(0.25)

    s([(h * 1.0).sum(), (h * 10.0).sum()], names=names, trunk=h)
    assert s._weights["main"] == pytest.approx(1.0)
    assert s._weights["loud"] == pytest.approx(DECAY * 0.25 + (1 - DECAY) * 0.1)


def test_cap_never_amplifies():
    h = torch.randn(8, requires_grad=True)
    s = _capped()
    for _ in range(5):
        s([(h * 5.0).sum(), (h * 0.01).sum()], names=["main", "quiet"], trunk=h)
    assert all(w <= 1.0 + 1e-9 for w in s._weights.values())
    assert s._weights["quiet"] == pytest.approx(1.0)


def test_cap_weights_carry_no_gradient():
    """A learned weight is a route for the model to switch off an objective it
    finds inconvenient. These are constants; there is nothing to learn."""
    assert list(_capped().parameters()) == []


def test_cap_falls_back_to_the_plain_sum_without_names():
    """The layer-wise fold calls strategies positionally and has no trunk."""
    losses = [torch.tensor(1.0), torch.tensor(2.0)]
    assert AnchorCapped()(losses).item() == pytest.approx(3.0)


def test_cap_is_registered_and_reports_its_weights():
    s = registry.lookup("strategies", "capped")()
    s.train()
    h = torch.randn(8, requires_grad=True)
    s([(h * 1.0).sum(), (h * 4.0).sum()], names=["main", "loud"], trunk=h)
    m = s.training_metrics()
    assert set(m) == {"blend_w_main", "blend_w_loud", "blend_w_min"}
    assert m["blend_w_min"] == pytest.approx(0.25)

    descs = blend_metric_descriptions(m.keys())
    assert set(descs) == set(m)
    # blend_w_min owns the chart; the per-term series ride it via series_group.
    assert "title" in descs["blend_w_min"]["chart"]
    for key in ("blend_w_main", "blend_w_loud"):
        assert "title" not in descs[key]["chart"]
        assert descs[key]["chart"]["series_group"] == "blend_w"


def test_cap_is_blind_to_a_value_neutral_surrogate():
    """``arm_surgery`` is ``g - g.detach()``: value identically 0, gradient the
    only task signal the trunk gets under prismatic9. A value-space balancer
    settles its weight at 1/L and diverges at L = 0; the cap reads the
    gradient, so the weight is the gradient ratio whatever the value."""
    h = torch.randn(8, requires_grad=True)
    g = (h * 3.0).sum()
    surrogate = g - g.detach()
    assert surrogate.item() == 0.0
    s = _capped()
    s([(h * 1.0).sum(), surrogate], names=["main", "arm_surgery"], trunk=h)
    assert s._weights["arm_surgery"] == pytest.approx(1.0 / 3.0)


def test_caps_survive_a_checkpoint_round_trip():
    h = torch.randn(8, requires_grad=True)
    a = _capped()
    a([(h * 1.0).sum(), (h * 4.0).sum()], names=["main", "loud"], trunk=h)

    b = AnchorCapped(interval=1)
    b.load_state_dict(a.state_dict())
    assert b._weights == a._weights
    assert b.anchor == a.anchor == "main"
