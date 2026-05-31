"""SQLite-based metrics logger for web visualization."""

import csv
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from praxis.metrics.training_metrics import metric_names


class MetricsLogger:
    """Logs training metrics to SQLite database for web visualization.

    This logger is framework-agnostic and can be used with any training loop.
    Metrics are written to a SQLite database with automatic deduplication.

    When the same step is logged multiple times (e.g., during gradient accumulation),
    the database merges the metrics, keeping the latest non-null values. This means
    you can log training metrics and validation metrics for the same step, and both
    will be preserved.

    Usage:
        logger = MetricsLogger(run_dir="build/runs/83492c812")
        logger.log(step=0, loss=2.45, lr=0.0003)
        logger.log(step=100, loss=2.12, val_loss=2.20)
        logger.close()

    Context manager usage:
        with MetricsLogger(run_dir="build/runs/83492c812") as logger:
            logger.log(step=0, loss=2.45, lr=0.0003)
    """

    # Native columns derive from the training-metric registry. Adding a
    # new scalar metric is a one-entry change in
    # ``praxis.metrics.training_metrics``; the column lands here
    # automatically and ``_ensure_columns_exist`` backfills old dbs.
    KNOWN_METRICS = metric_names()

    SCHEMA = (
        "CREATE TABLE IF NOT EXISTS metrics (\n"
        "    step INTEGER PRIMARY KEY,\n"
        "    ts REAL NOT NULL,\n"
        + "".join(f"    {col} REAL,\n" for col in KNOWN_METRICS)
        + "    extra_metrics TEXT\n"
        ");\n"
        "CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(ts);\n"
    )

    def __init__(
        self,
        run_dir: str,
        filename: str = "metrics.db",
        csv_mirror: bool = True,
        csv_interval_s: float = 30.0,
    ):
        """Initialize the metrics logger.

        Args:
            run_dir: Directory for the current run (e.g., "build/runs/83492c812")
            filename: Name of the database file (default: "metrics.db")
            csv_mirror: Also mirror the table to ``metrics.csv`` so tools that
                don't speak SQLite (LaTeX/pgfplots, the paper exporter) always
                have fresh data without the running web server.
            csv_interval_s: Minimum seconds between CSV rewrites; the mirror is
                also flushed on close, so a finished run is always complete.
        """
        self.run_dir = Path(run_dir)
        self.filepath = self.run_dir / filename
        self.lock = Lock()
        self._write_counter = 0
        self._commit_interval = 5  # Commit every 5 writes (optimized for small models)
        self._csv_mirror = csv_mirror
        self._csv_path = self.filepath.with_suffix(".csv")
        self._csv_interval = csv_interval_s
        self._csv_last = 0.0

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

        # Forward-compat: make sure any metric added after the table was
        # first created gets back-filled as a REAL column on an existing DB.
        self._ensure_columns_exist(self.KNOWN_METRICS)

    def _ensure_columns_exist(self, column_names):
        try:
            cursor = self.conn.cursor()
            cursor.execute("PRAGMA table_info(metrics)")
            existing = {row[1] for row in cursor.fetchall()}
            for col in column_names:
                if col not in existing:
                    self.conn.execute(f"ALTER TABLE metrics ADD COLUMN {col} REAL")
            self.conn.commit()
        except Exception as e:
            print(f"[MetricsLogger] Warning: Error ensuring columns exist: {e}")

    def log(self, step: int, **metrics: Any) -> None:
        """Log metrics for a training step.

        If metrics for this step already exist, they will be merged with new values.
        New non-null values overwrite existing values, but null values preserve existing data.

        Args:
            step: Training step number (required)
            **metrics: Arbitrary key-value metrics to log

        Examples:
            logger.log(step=100, loss=1.23, lr=3e-4)
            logger.log(step=200, loss=1.15, val_loss=1.20, perplexity=3.16)
        """
        with self.lock:
            # Separate known metrics from extra metrics
            known_values = {}
            extra_metrics = {}

            for key, value in metrics.items():
                if key in self.KNOWN_METRICS:
                    known_values[key] = value
                else:
                    extra_metrics[key] = value

            # Serialize extra metrics to JSON
            extra_json = (
                json.dumps(extra_metrics, separators=(",", ":"))
                if extra_metrics
                else None
            )

            # Build column lists for known metrics
            columns = ["step", "ts"] + self.KNOWN_METRICS + ["extra_metrics"]
            placeholders = ["?"] * len(columns)

            # Build values list
            values = [step, datetime.now().timestamp()]
            for metric in self.KNOWN_METRICS:
                values.append(known_values.get(metric))
            values.append(extra_json)

            # Build UPSERT query that merges values
            # COALESCE(excluded.loss, metrics.loss) means: use new value if non-null, else keep old
            update_clauses = ["ts = excluded.ts"]  # Always update timestamp
            for metric in self.KNOWN_METRICS:
                update_clauses.append(
                    f"{metric} = COALESCE(excluded.{metric}, metrics.{metric})"
                )
            update_clauses.append(
                "extra_metrics = COALESCE(excluded.extra_metrics, metrics.extra_metrics)"
            )

            query = f"""
                INSERT INTO metrics ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                ON CONFLICT(step) DO UPDATE SET
                {', '.join(update_clauses)}
            """

            self.conn.execute(query, values)

            # Commit strategy: commit frequently during early training for immediate dashboard feedback
            self._write_counter += 1
            # First 10 writes: commit every time (for immediate dashboard updates)
            if self._write_counter <= 10:
                self.conn.commit()
            # After that: commit every N writes to balance performance and freshness
            elif self._write_counter >= self._commit_interval:
                self.conn.commit()
                self._write_counter = 10  # Reset but keep the "past first 10" state

            # Mirror to CSV on an interval (lock already held).
            if (
                self._csv_mirror
                and time.monotonic() - self._csv_last >= self._csv_interval
            ):
                self._write_csv()

    def _write_csv(self) -> None:
        """Atomically mirror the metrics table to ``metrics.csv``.

        Assumes ``self.lock`` is held. A bad CSV write must never take down
        training, so failures are warned and swallowed.
        """
        try:
            cols = ["step", "ts"] + self.KNOWN_METRICS
            rows = self.conn.execute(
                f"SELECT {', '.join(cols)} FROM metrics ORDER BY step"
            ).fetchall()
            tmp = self._csv_path.with_suffix(".csv.tmp")
            with open(tmp, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(cols)
                writer.writerows(rows)
            os.replace(tmp, self._csv_path)  # atomic; readers never see a partial file
            self._csv_last = time.monotonic()
        except Exception as e:
            print(f"[MetricsLogger] Warning: CSV mirror failed: {e}")

    def close(self) -> None:
        """Close the database connection."""
        with self.lock:
            if self.conn:
                # Commit any pending writes before closing
                if self._write_counter > 0:
                    self.conn.commit()
                if self._csv_mirror:
                    self._write_csv()  # final flush: a finished run's CSV is complete
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
