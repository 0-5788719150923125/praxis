"""Tests for praxis/policies/loop_modes.py: the Loop UI's parse and score
contracts."""

from praxis import registry


class TestLoopModes:
    """The loop-mode registry: parse + score contracts."""

    def test_registry_and_default(self):
        from praxis.policies.loop_modes import get_loop_mode

        modes = registry.namespace("loop_modes")
        assert set(modes) >= {"calibration", "approval"}
        assert all(get_loop_mode(k).name == k for k in modes)
        assert get_loop_mode().name == "calibration"
        assert get_loop_mode("approval").name == "approval"
        assert get_loop_mode("nonsense").name == "calibration"  # safe fallback

    def test_calibration_parse(self):
        from praxis.policies.loop_modes import get_loop_mode

        m = get_loop_mode("calibration")
        assert m.parse("A pun!\n+0.6") == ("A pun!", 0.6)
        # No trailing score line -> the whole reply is content.
        assert m.parse("just a joke") == ("just a joke", None)
        # A bare number far outside the slider range is content (e.g. a year),
        # not a wildly-clamped prediction.
        assert m.parse("In the year\n1942") == ("In the year\n1942", None)

    def test_calibration_score_rewards_small_corrections(self):
        from praxis.policies.loop_modes import get_loop_mode

        m = get_loop_mode("calibration")
        perfect = m.score(0.6, 0.6)
        wrong = m.score(-0.4, 0.6)
        assert perfect["activation"] == 1.0
        assert perfect["extra"]["correction"] == 0.0
        assert wrong["activation"] == 0.5  # 1 - |corr|/2 = 1 - 0.5
        assert wrong["reward"] == -0.4  # valence stays the user's signed score
        # No parseable prediction -> degrade to approval semantics.
        assert m.score(1.0, None)["activation"] == 1.0
        assert m.score(1.0, None)["extra"] == {}
