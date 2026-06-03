"""Process-global channel for live engagement-prediction rewards (PLAN.md P4/P5).

The `Print` UI computes a reward from a real user's response to a model-led
question and submits it here; the training side drains it into the RL controller's
reward buffer. This decouples the sparse, asynchronous UI events from the
training-loop cadence - the environment-level hook that makes online learning
possible. Thread-safe; lives outside any model so the web thread and the trainer
can both reach it.
"""

import threading
from collections import deque

from praxis.policies.engagement_reward import HomeostaticEnergy, activation, recall


class LiveEngagementChannel:
    def __init__(self, maxlen: int = 256):
        self._lock = threading.Lock()
        self._buffer = deque(maxlen=maxlen)  # undrained live rewards
        self._energy = HomeostaticEnergy()
        self._count = 0
        self._last = None

    def submit(self, predicted_tokens, response_tokens) -> dict:
        """Score one real interaction, fold it into the live energy, and buffer
        it for the trainer to drain. Returns the event."""
        a = activation(predicted_tokens, response_tokens)
        r = recall(predicted_tokens, response_tokens)
        with self._lock:
            energy = self._energy.update(a)
            event = {"activation": a, "recall": r, "reward": r, "energy": energy}
            self._buffer.append(event)
            self._count += 1
            self._last = event
        return event

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "energy": self._energy.value,
                "count": self._count,
                "buffered": len(self._buffer),
                "last": self._last,
            }

    def drain(self) -> list:
        """Pop all buffered live rewards (for the RL controller to consume)."""
        with self._lock:
            items = list(self._buffer)
            self._buffer.clear()
            return items


# The single shared channel. Import this, don't instantiate your own.
LIVE_ENGAGEMENT = LiveEngagementChannel()
