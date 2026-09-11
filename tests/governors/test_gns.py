"""GNS governor math (praxis/governors/gns.py): the noise-scale estimator and the tier controller."""

import math

import torch

from praxis.governors.gns import BatchTierController, GradientNoiseEstimator

# ── estimator ────────────────────────────────────────────────────────────


def test_estimator_recovers_known_noise_scale():
    """Feed squared norms of synthetic small/big-batch mean gradients whose
    true B_noise = tr(cov)/|mu|^2 is known; the EMA ratio must land near it."""
    torch.manual_seed(0)
    d, sigma, b_small, b_big = 512, 0.5, 4, 64
    mu = torch.randn(d) / math.sqrt(d)  # |mu|^2 ~ 1
    true_noise = d * sigma**2 / float(mu.pow(2).sum())
    est = GradientNoiseEstimator()
    for _ in range(600):
        g_small = mu + sigma * torch.randn(d) / math.sqrt(b_small)
        g_big = mu + sigma * torch.randn(d) / math.sqrt(b_big)
        est.update(
            small_sq=float(g_small.pow(2).sum()),
            big_sq=float(g_big.pow(2).sum()),
            b_small=b_small,
            b_big=b_big,
        )
    measured = est.noise_scale()
    assert measured is not None
    assert abs(measured - true_noise) / true_noise < 0.25


def test_estimator_not_ready_early_and_rejects_negative_signal():
    est = GradientNoiseEstimator()
    assert est.noise_scale() is None
    # small_sq >> big_sq at these batch sizes drives the |G|^2 estimate
    # negative: (2*1 - 1*10)/1 = -8. Must refuse to report a scale.
    for _ in range(est.min_updates):
        est.update(small_sq=10.0, big_sq=1.0, b_small=1, b_big=2)
    assert est.ready
    assert est.noise_scale() is None


def test_estimator_single_point_is_a_noop():
    est = GradientNoiseEstimator()
    est.update(small_sq=1.0, big_sq=1.0, b_small=16, b_big=16)
    assert est._updates == 0


# ── tier controller ──────────────────────────────────────────────────────


def test_controller_moves_one_tier_with_deadband_and_clamps():
    ctl = BatchTierController(max_rows=512)
    # Far above: one doubling per decision, never a jump.
    assert ctl.desired_rows(32, 1584.0) == 64
    # Inside the deadband (log2(70/64) ~ 0.13): hold.
    assert ctl.desired_rows(64, 70.0) == 64
    # Far below: halve.
    assert ctl.desired_rows(64, 10.0) == 32
    # Ceiling and floor.
    assert ctl.desired_rows(512, 1e9) == 512
    assert ctl.desired_rows(2, 0.001) == 2
    # No measurement: hold.
    assert ctl.desired_rows(128, None) == 128


def test_controller_floor_is_two_rows_not_a_microbatch():
    """The floor exists to keep the two-point estimator alive (one row per
    microbatch at the two-microbatch minimum), NOT to respect batch_size -
    which is a per-microbatch ceiling and must not bound the batch below."""
    ctl = BatchTierController(max_rows=512)
    assert ctl.min_rows == 2
    rows = 512
    for _ in range(20):  # a sustained tiny noise scale must descend all the way
        rows = ctl.desired_rows(rows, 1.0)
    assert rows == 2


def test_controller_regulates_against_delivered_rows():
    """When a long-sequence cycle's attention budget caps rows below the
    target, the decision must use what ran - not what was asked for."""
    ctl = BatchTierController(max_rows=512)
    # Target 128, but only 32 rows actually landed; B_noise of 100 is ABOVE
    # what ran, so the honest move is up even though 100 < 128.
    assert ctl.desired_rows(128, 100.0, measured=32.0) == 256
    # Same noise judged against the requested target would have held.
    assert ctl.desired_rows(128, 100.0) == 128


def test_controller_hysteresis_prevents_immediate_flap():
    """Moving a tier shifts the reference an octave, so net hysteresis is
    2*deadband - 1. A measurement that just triggered an up-move must not be
    able to trigger the down-move from the new tier."""
    ctl = BatchTierController(max_rows=512)
    # Up-threshold of the 32-row tier is the next tier (64 rows).
    assert ctl.desired_rows(32, 63.0) == 32
    assert ctl.desired_rows(32, 65.0) == 64
    # The same measurement holds at 64 rows...
    assert ctl.desired_rows(64, 65.0) == 64
    # ...as does anything above 64's down-threshold (32 rows).
    assert ctl.desired_rows(64, 33.0) == 64
    assert ctl.desired_rows(64, 31.0) == 32
