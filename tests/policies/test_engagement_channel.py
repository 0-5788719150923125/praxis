"""Tests for praxis/policies/engagement_channel.py: the live reward buffer."""

from praxis.policies.engagement_channel import LIVE_ENGAGEMENT


def test_channel_buffers_for_trainer_drain():
    LIVE_ENGAGEMENT.drain()  # a process-wide singleton
    before = LIVE_ENGAGEMENT.snapshot()["count"]
    LIVE_ENGAGEMENT.submit(["paris"], ["paris", "i", "think"])
    assert LIVE_ENGAGEMENT.snapshot()["count"] == before + 1
    drained = LIVE_ENGAGEMENT.drain()
    assert drained and drained[-1]["recall"] == 1.0
    assert LIVE_ENGAGEMENT.snapshot()["buffered"] == 0
