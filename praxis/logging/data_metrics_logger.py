"""Framework-agnostic data metrics logger for preprocessing and sampling visualization."""

import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


class DataMetricsLogger:
    """Logs data preprocessing and sampling metrics to JSONL file for web visualization.

    This logger tracks metrics related to data processing, such as:
    - Document sampling weights (for dynamic weighting strategies)
    - Dataset statistics and distributions
    - Data preprocessing timings
    - Batch composition metrics

    Metrics are written to a JSONL file (one JSON object per line) for efficient
    append-only writes and incremental reading.

    Usage:
        logger = DataMetricsLogger(run_dir="build/runs/83492c812")
        logger.log(step=0, sampling_weights={"doc1": 0.5, "doc2": 0.5})
        logger.log(step=100, sampling_weights={"doc1": 0.3, "doc2": 0.7})
        logger.close()

    Context manager usage:
        with DataMetricsLogger(run_dir="build/runs/83492c812") as logger:
            logger.log(step=0, sampling_weights={"doc1": 0.5, "doc2": 0.5})
    """

    def __init__(self, run_dir: str, filename: str = "data_metrics.jsonl"):
        """Initialize the data metrics logger.

        Args:
            run_dir: Directory for the current run (e.g., "build/runs/83492c812")
            filename: Name of the metrics file (default: "data_metrics.jsonl")
        """
        self.run_dir = Path(run_dir)
        self.filepath = self.run_dir / filename
        self.lock = Lock()
        self._file_handle = None

        # Ensure directory exists
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        self._file_handle = open(self.filepath, "a", buffering=1)  # Line buffered

    def log(self, step: int, **metrics: Any) -> None:
        """Log data metrics for a training step.

        Args:
            step: Training step number (required)
            **metrics: Arbitrary key-value metrics to log
                Common keys:
                - sampling_weights: Dict[str, float] - Document sampling weights
                - dataset_stats: Dict - Dataset statistics
                - preprocessing_time: float - Time spent preprocessing

        Examples:
            logger.log(step=100, sampling_weights={"arxiv": 0.3, "wiki": 0.7})
            logger.log(step=200, sampling_weights={"arxiv": 0.4, "wiki": 0.6},
                      dataset_stats={"total_docs": 1000})
        """
        with self.lock:
            if self._file_handle is None or self._file_handle.closed:
                # Reopen if closed
                self._file_handle = open(self.filepath, "a", buffering=1)

            # Build log entry
            entry = {"step": step, "ts": datetime.now().isoformat(), **metrics}

            # Write as single line
            json.dump(entry, self._file_handle, separators=(",", ":"))
            self._file_handle.write("\n")
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
