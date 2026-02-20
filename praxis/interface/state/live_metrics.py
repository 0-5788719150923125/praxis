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
        self.info_dict = {}
        self.log_lines = deque(maxlen=200)
        self._log_lock = threading.Lock()
        self._update_count = 0

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

    def snapshot(self):
        """Return a JSON-serializable snapshot of current metrics."""
        with self.state.lock:
            return {
                "loss": (
                    self.state.train_losses[-1]
                    if self.state.train_losses
                    else None
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
                "local_experts": self.state.local_experts,
                "remote_experts": self.state.remote_experts,
                "mode": self.state.mode,
                "seed": self.state.seed,
                "arg_hash": self.state.arg_hash,
                "url": self.state.url,
                "hours_elapsed": self.state.hours_since(),
                "status_text": self.status_text,
                "info": self.info_dict,
                "update_count": self._update_count,
                "log_lines": list(self.log_lines)[-50:],
            }
