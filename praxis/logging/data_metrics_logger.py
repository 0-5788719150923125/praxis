"""SQLite-based data metrics logger for preprocessing and sampling visualization."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any


class DataMetricsLogger:
    """Logs data preprocessing and sampling metrics to SQLite database for web visualization.

    This logger tracks metrics related to data processing, such as:
    - Document sampling weights (for dynamic weighting strategies)
    - Dataset statistics and distributions
    - Data preprocessing timings
    - Batch composition metrics

    Metrics are written to a SQLite database with automatic deduplication.

    Usage:
        logger = DataMetricsLogger(run_dir="build/runs/83492c812")
        logger.log(step=0, sampling_weights={"doc1": 0.5, "doc2": 0.5})
        logger.log(step=100, sampling_weights={"doc1": 0.3, "doc2": 0.7})
        logger.close()

    Context manager usage:
        with DataMetricsLogger(run_dir="build/runs/83492c812") as logger:
            logger.log(step=0, sampling_weights={"doc1": 0.5, "doc2": 0.5})
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS data_metrics (
        step INTEGER PRIMARY KEY,
        ts REAL NOT NULL,
        sampling_weights TEXT,
        dataset_stats TEXT,
        extra_metrics TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_data_metrics_ts ON data_metrics(ts);
    """

    def __init__(self, run_dir: str, filename: str = "data_metrics.db"):
        """Initialize the data metrics logger.

        Args:
            run_dir: Directory for the current run (e.g., "build/runs/83492c812")
            filename: Name of the database file (default: "data_metrics.db")
        """
        self.run_dir = Path(run_dir)
        self.filepath = self.run_dir / filename
        self.lock = Lock()
        self._write_counter = 0
        self._commit_interval = 10  # Commit every 10 writes (data metrics are less frequent)

        # Ensure directory exists
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Open database connection
        self.conn = sqlite3.connect(
            self.filepath, check_same_thread=False, timeout=30.0
        )

        # Enable WAL mode for concurrent reads during training
        self.conn.execute("PRAGMA journal_mode=WAL")
        # Use NORMAL synchronous mode for better write performance
        self.conn.execute("PRAGMA synchronous=NORMAL")

        # Create schema
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

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
            # Extract known data metric types
            sampling_weights = metrics.pop("sampling_weights", None)
            dataset_stats = metrics.pop("dataset_stats", None)

            # Remaining metrics go to extra
            extra_metrics = metrics if metrics else None

            # Serialize to JSON
            sampling_weights_json = (
                json.dumps(sampling_weights, separators=(",", ":"))
                if sampling_weights
                else None
            )
            dataset_stats_json = (
                json.dumps(dataset_stats, separators=(",", ":"))
                if dataset_stats
                else None
            )
            extra_json = (
                json.dumps(extra_metrics, separators=(",", ":"))
                if extra_metrics
                else None
            )

            # UPSERT query with COALESCE to merge values
            query = """
                INSERT INTO data_metrics (step, ts, sampling_weights, dataset_stats, extra_metrics)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(step) DO UPDATE SET
                ts = excluded.ts,
                sampling_weights = COALESCE(excluded.sampling_weights, data_metrics.sampling_weights),
                dataset_stats = COALESCE(excluded.dataset_stats, data_metrics.dataset_stats),
                extra_metrics = COALESCE(excluded.extra_metrics, data_metrics.extra_metrics)
            """

            self.conn.execute(
                query,
                (
                    step,
                    datetime.now().timestamp(),
                    sampling_weights_json,
                    dataset_stats_json,
                    extra_json,
                ),
            )

            # Commit only every N writes to reduce WAL bloat
            self._write_counter += 1
            if self._write_counter >= self._commit_interval:
                self.conn.commit()
                self._write_counter = 0

    def close(self) -> None:
        """Close the database connection."""
        with self.lock:
            if self.conn:
                # Commit any pending writes before closing
                if self._write_counter > 0:
                    self.conn.commit()
                self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup
