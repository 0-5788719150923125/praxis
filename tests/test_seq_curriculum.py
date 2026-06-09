"""Adaptive sequence-length curriculum (learning-progress bandit over tiers).

The controller should sample more of the multiplier the model is improving
fastest on, keep every arm alive (exploration floor), respect tier eligibility
(batch_size >= m**2), and stay inert until enabled.
"""

import random

from praxis.data.datasets.manager import (
    SEQUENCE_MULTIPLIER_TIERS,
    sample_sequence_multiplier,
)
from praxis.data.seq_curriculum import SequenceCurriculum as SC

TIERS = ((8, 0.001), (4, 0.01), (2, 0.1))


def _reset():
    SC.reset()


def test_disabled_falls_back_to_fixed_roll():
    _reset()
    # Not enabled: sample() returns None, so the fixed roll runs and always
    # yields an eligible multiplier.
    assert SC.sample(128, TIERS, random.Random(0)) is None
    got = {sample_sequence_multiplier(128, TIERS, random) for _ in range(200)}
    assert got <= {1, 2, 4, 8}


def test_cold_start_before_first_progress_is_fixed():
    _reset()
    SC.enable(block_size=10, tiers=TIERS)
    # Armed but no observations yet -> no distribution -> caller falls back.
    assert SC.sample(128, TIERS, random.Random(0)) is None
    _reset()


def test_upsamples_the_multiplier_with_most_learning_progress():
    _reset()
    SC.enable(block_size=10, tiers=TIERS)
    loss = {1: 5.0, 2: 5.0, 4: 5.0, 8: 5.0}
    for _ in range(40):
        for m in (1, 2, 4, 8):
            if m == 4:
                loss[4] -= 0.05  # only m=4 keeps improving
            SC.observe(10 * m, loss[m])
    probs = SC.shared_probs
    assert probs[4] == max(probs.values())  # most progress -> most sampled
    assert min(probs.values()) > 0  # exploration floor: no arm starves
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    _reset()


def test_progress_not_raw_loss_drives_it():
    # A multiplier with low absolute loss but NO improvement must not win over
    # one that is actively improving - guards against the long-sequence bias.
    _reset()
    SC.enable(block_size=10, tiers=TIERS)
    flat_low = 1.0  # m=8 sits at low loss but never improves
    improving = 5.0  # m=2 starts high but keeps dropping
    for _ in range(40):
        SC.observe(80, flat_low)  # m=8, flat
        improving -= 0.05
        SC.observe(20, improving)  # m=2, improving
        SC.observe(10, 5.0)  # m=1 flat
        SC.observe(40, 5.0)  # m=4 flat
    probs = SC.shared_probs
    assert probs[2] > probs[8]  # improvement beats a low-but-static loss
    _reset()


def test_eligibility_respected():
    _reset()
    SC.enable(block_size=10, tiers=TIERS)
    for _ in range(5):
        for m in (1, 2, 4, 8):
            SC.observe(10 * m, 5.0 - 0.01 * m)
    rng = random.Random(1)
    assert {SC.sample(1, TIERS, rng) for _ in range(50)} == {1}  # only m=1 fits
    assert {SC.sample(4, TIERS, rng) for _ in range(100)} <= {1, 2}  # m<=2 fits
    _reset()


def test_metrics_exposed_when_active():
    _reset()
    assert SC.metrics() == {}  # nothing when disabled
    SC.enable(block_size=10, tiers=TIERS)
    for _ in range(3):
        for m in (1, 2, 4, 8):
            SC.observe(10 * m, 4.0)
    mt = SC.metrics()
    assert all(f"seq_prob_x{m}" in mt for m in (1, 2, 4, 8))
    assert all(f"seq_progress_x{m}" in mt for m in (1, 2, 4, 8))
    _reset()
