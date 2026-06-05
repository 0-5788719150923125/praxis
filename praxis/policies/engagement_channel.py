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

from praxis.policies.engagement_reward import (
    HomeostaticEnergy,
    activation,
    recall,
    response_energy,
)


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
        r = recall(predicted_tokens, response_tokens)
        # Any genuine answer sustains the energy (and shifts the RL baseline via
        # ingest_live), with recall lifting it toward 1.0. The raw prediction
        # match is kept only as a quality metric.
        a = response_energy(bool(response_tokens), r)
        with self._lock:
            energy = self._energy.update(a)
            event = {
                "activation": a,
                "match": activation(predicted_tokens, response_tokens),
                "recall": r,
                "reward": r,
                "energy": energy,
            }
            self._buffer.append(event)
            self._count += 1
            self._last = event
        return event

    def submit_scalar(self, activation, reward=None) -> dict:
        """Fold a direct activation in [0,1] into the energy and buffer it, with an
        optional signed ``reward`` (e.g. a -1..1 want->need slider score) recorded
        alongside. ``activation`` drives the homeostatic energy; ``reward`` is the
        logged learning signal. Like ``submit`` but the score is given outright
        rather than computed from token overlap."""
        q = max(0.0, min(1.0, float(activation)))
        rw = q if reward is None else float(reward)
        # A human bothering to score sustains the energy on its own, with the
        # score lifting it toward 1.0. Valence stays in ``reward`` (the signed
        # want->need learning signal); energy is sustenance, not sign.
        a = response_energy(True, q)
        with self._lock:
            energy = self._energy.update(a)
            event = {"activation": a, "recall": rw, "reward": rw, "energy": energy}
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


# Shared process-global channels. Import these, don't instantiate your own.
# LIVE_ENGAGEMENT: Print question/answer recall. LIVE_JOKES: joke approvals.
LIVE_ENGAGEMENT = LiveEngagementChannel()
LIVE_JOKES = LiveEngagementChannel()
