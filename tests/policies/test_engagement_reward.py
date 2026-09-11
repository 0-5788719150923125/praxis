"""Tests for praxis/policies/engagement_reward.py: recall rewards and the
homeostatic energy."""

from praxis.policies.engagement_reward import (
    ENERGY_FLOOR,
    HomeostaticEnergy,
    activation,
    recall,
    response_energy,
)


class TestReward:
    def test_activation_fires_on_overlap(self):
        assert activation([1, 2, 3], [3, 9]) == 1.0
        assert activation([1, 2, 3], [7, 8]) == 0.0
        assert activation([], [1]) == 0.0

    def test_recall_is_graded(self):
        assert recall([1, 2, 3, 4], [1, 2, 9]) == 0.5  # 2 of 4 predicted mentioned
        assert recall([1, 2], [1, 2]) == 1.0
        assert recall([], [1]) == 0.0

    def test_response_energy_floors_on_engagement(self):
        # Any genuine response sustains energy regardless of prediction match;
        # quality lifts it to 1.0. No interaction -> no energy.
        assert response_energy(False) == 0.0
        assert response_energy(True, 0.0) == ENERGY_FLOOR
        assert response_energy(True, 1.0) == 1.0
        assert ENERGY_FLOOR < response_energy(True, 0.5) < 1.0


class TestHomeostaticEnergy:
    def test_accumulates_fast_then_satiates(self):
        e = HomeostaticEnergy()
        first = e.update(1.0)
        # Diminishing returns: each successive full activation adds less.
        gains = [first]
        for _ in range(5):
            before = e.value
            e.update(1.0)
            gains.append(e.value - before)
        assert gains[0] > 0
        assert all(gains[i] >= gains[i + 1] - 1e-9 for i in range(len(gains) - 1))

    def test_decays_without_activation(self):
        # Decay is wall-clock, not event-driven: energy depletes overnight even
        # if nothing ever calls update().
        t = [0.0]
        e = HomeostaticEnergy(init=0.5, clock=lambda: t[0])
        t[0] += e.half_life_s
        assert abs(e.value - 0.25) < 1e-9
        t[0] += 8 * e.half_life_s
        assert e.value < 0.01

    def test_energy_stays_bounded(self):
        e = HomeostaticEnergy()
        for _ in range(10_000):
            e.update(1.0)
        assert 0.0 <= e.value <= e.e_max
