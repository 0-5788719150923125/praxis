"""Framework-agnostic metrics logger for web visualization."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from threading import Lock


class MetricsLogger:
    """Logs training metrics to JSONL file for web visualization.

    This logger is framework-agnostic and can be used with any training loop.
    Metrics are written to a JSONL file (one JSON object per line) for efficient
    append-only writes and incremental reading.

    Usage:
        logger = MetricsLogger(run_dir="build/runs/83492c812")
        logger.log(step=0, loss=2.45, lr=0.0003)
        logger.log(step=100, loss=2.12, val_loss=2.20)
        logger.close()

    Context manager usage:
        with MetricsLogger(run_dir="build/runs/83492c812") as logger:
            logger.log(step=0, loss=2.45, lr=0.0003)
    """

    def __init__(self, run_dir: str, filename: str = "metrics.jsonl"):
        """Initialize the metrics logger.

        Args:
            run_dir: Directory for the current run (e.g., "build/runs/83492c812")
            filename: Name of the metrics file (default: "metrics.jsonl")
        """
        self.run_dir = Path(run_dir)
        self.filepath = self.run_dir / filename
        self.lock = Lock()
        self._file_handle = None

        # Ensure directory exists
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        self._file_handle = open(self.filepath, 'a', buffering=1)  # Line buffered

    def log(self, step: int, **metrics: Any) -> None:
        """Log metrics for a training step.

        Args:
            step: Training step number (required)
            **metrics: Arbitrary key-value metrics to log

        Examples:
            logger.log(step=100, loss=1.23, lr=3e-4)
            logger.log(step=200, loss=1.15, val_loss=1.20, perplexity=3.16)
        """
        with self.lock:
            if self._file_handle is None or self._file_handle.closed:
                # Reopen if closed
                self._file_handle = open(self.filepath, 'a', buffering=1)

            # Build log entry
            entry = {
                "step": step,
                "ts": datetime.now().isoformat(),
                **metrics
            }

            # Write as single line
            json.dump(entry, self._file_handle, separators=(',', ':'))
            self._file_handle.write('\n')
            # Note: buffering=1 means line-buffered, so flush happens automatically

    def close(self) -> None:
        """Close the file handle."""
        with self.lock:
            if self._file_handle and not self._file_handle.closed:
                self._file_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
