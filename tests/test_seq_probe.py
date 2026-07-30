"""Probe-attribution sequence curriculum (praxis/data/seq_probe.py).

The invariants that matter are the ones the previous controller failed:

- an arm's coefficient must recover its true value from the regression,
- an arm with no measurable edge must not be handed a confident share,
- the fit must track a change in which arm is best rather than average over all
  of history,
- the fixed per-tier roll must remain the cold-start path.

The controller this replaced (a learning-progress bandit scoring each arm by the
loss drop between two visits to it) is gone rather than deprecated: that drop
measures how much the WHOLE model improved in the interval, so it carried no
information about arm quality - a worthless arm still earned a full share, and
sampling an arm more often shortened its own interval, making the mechanism
negative feedback on visit rate that drove the mix to uniform.
"""

import random

import pytest

from praxis.data.seq_probe import SequenceProbe

TIERS = ((4, 0.01), (2, 0.1))
ARMS = [1, 2, 4]


@pytest.fixture(autouse=True)
def _clean():
    SequenceProbe.reset()
    yield
    SequenceProbe.reset()


def feed(values, windows=400, noise=5.0, seed=0, max_visits=40):
    """Run windows whose probe delta is a linear function of the arm counts."""
    rng = random.Random(seed)
    for _ in range(windows):
        visits = {m: rng.randint(0, max_visits) for m in ARMS}
        delta = sum(values[m] * c for m, c in visits.items()) + rng.gauss(0.0, noise)
        SequenceProbe.observe_window(visits, delta)
    return dict(zip(SequenceProbe.arms, SequenceProbe._beta))


# ── the regression ───────────────────────────────────────────────────────


def test_recovers_known_arm_values():
    SequenceProbe.enable(64, TIERS)
    beta = feed({1: 1.0, 2: 3.0, 4: 0.0})
    assert beta[1] == pytest.approx(1.0, abs=0.15)
    assert beta[2] == pytest.approx(3.0, abs=0.15)
    assert beta[4] == pytest.approx(0.0, abs=0.15)


def test_best_arm_gets_the_mass():
    SequenceProbe.enable(64, TIERS)
    feed({1: 1.0, 2: 3.0, 4: 0.0})
    probs = SequenceProbe.shared_probs
    assert probs[2] > 0.8
    assert probs[2] > probs[1] > probs[4] or probs[2] > probs[4]


def test_worthless_arm_is_starved():
    """The decisive failure of the learning-progress bandit: an arm with zero
    teaching value still earned a full 25% share."""
    SequenceProbe.enable(64, TIERS)
    feed({1: 2.0, 2: 2.0, 4: 0.0})
    probs = SequenceProbe.shared_probs
    assert probs[4] < 0.15, probs
    assert probs[1] + probs[2] > 0.8, probs


def test_no_signal_is_not_certainty():
    """Pure noise must not produce a near-1.0 share. The old z-scoring divided
    by the spread BETWEEN arms, which collapses when no arm is better, so noise
    rendered as confident."""
    SequenceProbe.enable(64, TIERS)
    feed({1: 0.0, 2: 0.0, 4: 0.0})
    probs = SequenceProbe.shared_probs
    assert max(probs.values()) < 0.85, probs
    # And no arm may be effectively excluded on no evidence.
    assert min(probs.values()) > 0.02, probs


def test_no_signal_leaves_every_t_statistic_insignificant():
    """The honest read on 'is there anything to exploit' is the t-statistic
    card, not the mix: under pure noise every |t| stays small."""
    SequenceProbe.enable(64, TIERS)
    feed({1: 0.0, 2: 0.0, 4: 0.0})
    assert all(abs(t) < 2.5 for t in SequenceProbe._tstat), SequenceProbe._tstat


def test_real_signal_is_significant():
    SequenceProbe.enable(64, TIERS)
    feed({1: 1.0, 2: 3.0, 4: 0.0})
    by_arm = dict(zip(SequenceProbe.arms, SequenceProbe._tstat))
    assert by_arm[2] > 3.0, by_arm


def test_fit_tracks_a_change_in_the_best_arm():
    """Forgetting is what makes this a controller rather than a historian."""
    SequenceProbe.enable(64, TIERS)
    feed({1: 3.0, 2: 0.0, 4: 0.0}, windows=300, seed=1)
    assert SequenceProbe.shared_probs[1] > 0.6
    feed({1: 0.0, 2: 3.0, 4: 0.0}, windows=300, seed=2)
    assert SequenceProbe.shared_probs[2] > 0.6, SequenceProbe.shared_probs


# ── sampling contract ────────────────────────────────────────────────────


def test_cold_start_defers_to_the_fixed_roll():
    """Until the fit is trusted, sample() returns None so
    sample_sequence_multiplier falls through to the per-tier chances."""
    SequenceProbe.enable(64, TIERS)
    rng = random.Random(0)
    assert SequenceProbe.sample(64, TIERS, rng) is None
    feed({1: 1.0, 2: 3.0, 4: 0.0}, windows=SequenceProbe.min_windows - 1)
    assert SequenceProbe.sample(64, TIERS, rng) is None
    feed({1: 1.0, 2: 3.0, 4: 0.0}, windows=50)
    assert SequenceProbe.sample(64, TIERS, rng) in ARMS


def test_disabled_controller_is_inert():
    rng = random.Random(0)
    assert SequenceProbe.sample(64, TIERS, rng) is None
    assert SequenceProbe.metrics() == {}
    SequenceProbe.observe_window({1: 10}, 0.5)  # must not raise
    assert SequenceProbe.shared_probs is None


def test_sampling_respects_the_row_budget():
    """The m**2 trade means a length is only affordable when the row budget has
    m**2 rows to give up - the same eligibility rule the batch schedule uses."""
    SequenceProbe.enable(64, TIERS)
    feed({1: 1.0, 2: 3.0, 4: 0.0}, windows=50)
    rng = random.Random(0)
    drawn = {SequenceProbe.sample(4, TIERS, rng) for _ in range(200)}
    assert drawn <= {1, 2}, drawn
    drawn = {SequenceProbe.sample(2, TIERS, rng) for _ in range(200)}
    assert drawn == {1}, drawn


def test_enable_is_idempotent_so_a_resume_keeps_its_fit():
    """Lightning restores callback state BEFORE on_train_start, so re-arming
    must not wipe the restored regression."""
    SequenceProbe.enable(64, TIERS)
    feed({1: 1.0, 2: 3.0, 4: 0.0}, windows=50)
    before = list(SequenceProbe._beta)
    windows = SequenceProbe._windows
    SequenceProbe.enable(64, TIERS)
    assert SequenceProbe._beta == before
    assert SequenceProbe._windows == windows


def test_metrics_expose_value_and_evidence():
    SequenceProbe.enable(64, TIERS)
    feed({1: 1.0, 2: 3.0, 4: 0.0}, windows=50)
    m = SequenceProbe.metrics()
    for arm in ARMS:
        assert f"seq_prob_x{arm}" in m
        assert f"seq_value_x{arm}" in m
        assert f"seq_tstat_x{arm}" in m
    assert sum(m[f"seq_prob_x{a}"] for a in ARMS) == pytest.approx(1.0, abs=1e-6)


def test_metric_descriptions_fold_in_when_armed():
    from praxis.metrics.descriptions import get_metric_descriptions

    class _Bare:
        pass

    plain = _Bare()
    assert "seq_tstat_x2" not in get_metric_descriptions(plain)

    armed = _Bare()
    armed._seq_probe_metrics = {"seq_prob_x1": 0.5}
    descs = get_metric_descriptions(armed)
    assert descs["seq_tstat_x2"]["caller"] == "SequenceProbe"
    assert descs["seq_value_x2"]["chart"]["series_group"] == "seq_value"
