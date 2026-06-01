"""Singleton live metrics holder for real-time web streaming.

Holds a MetricsState instance independently of TerminalDashboard,
so metrics can be streamed to web clients even in headless mode.
"""

import threading
from collections import deque

from .metrics import MetricsState


class LiveMetrics:
    """Singleton that holds real-time metrics independently of any dashboard."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.state = MetricsState(max_data_points=1000)
        self.status_text = "_initializing"
        self.contexts = (
            []
        )  # rolling context blocks: [{name, description, temperature, chance, text}]
        self.info_dict = {}
        self.log_lines = deque(maxlen=200)
        self._log_lock = threading.Lock()
        self._update_count = 0
        # Discrete backend events (stage transitions, milestones, ...) streamed
        # to the web app's notification bell. Each carries a monotonic id so the
        # frontend can dedupe across the repeated snapshot stream.
        self.events = deque(maxlen=100)
        self._event_seq = 0
        self._event_lock = threading.Lock()

    def add_log(self, message):
        """Add a log line."""
        with self._log_lock:
            if not message:
                return
            stripped = message.rstrip()
            lines = stripped.split("\n")
            for line in lines:
                if line.strip():
                    self.log_lines.append(line)

    def add_event(self, message, level="info"):
        """Record a discrete event for the web notification feed.

        Generic entry point: any backend code holding the LiveMetrics singleton
        can announce a milestone (stage transition, checkpoint, warning). Bumps
        the update count so the websocket emitter pushes it promptly.
        """
        if not message:
            return
        with self._event_lock:
            self._event_seq += 1
            self.events.append(
                {
                    "id": self._event_seq,
                    "message": str(message),
                    "level": level,
                    "stage": self.state.stage,
                    "hours_elapsed": self.state.hours_since(),
                }
            )
            self._update_count += 1

    def snapshot(self):
        """Return a JSON-serializable snapshot of current metrics."""
        with self.state.lock:
            return {
                "loss": (
                    self.state.train_losses[-1] if self.state.train_losses else None
                ),
                "loss_history": list(self.state.train_losses)[-50:],
                "val_loss": self.state.val_loss,
                "accuracy": self.state.accuracy,
                "fitness": self.state.fitness,
                "memory_churn": self.state.memory_churn,
                "batch": self.state.batch,
                "step": self.state.step,
                "rate": self.state.rate,
                "num_tokens": self.state.num_tokens,
                "context_tokens": self.state.context_tokens,
                "total_params": self.state.total_params,
                "local_layers": self.state.local_layers,
                "remote_layers": self.state.remote_layers,
                "mode": self.state.mode,
                "stage": self.state.stage,
                "events": list(self.events),
                "seed": self.state.seed,
                "arg_hash": self.state.arg_hash,
                "url": self.state.url,
                "hours_elapsed": self.state.hours_since(),
                "status_text": self.status_text,
                "contexts": list(self.contexts),
                "info": self.info_dict,
                "update_count": self._update_count,
                "log_lines": list(self.log_lines)[-50:],
            }
