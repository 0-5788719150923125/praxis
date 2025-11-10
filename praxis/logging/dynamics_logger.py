"""SQLite-based dynamics logger for gradient visualization."""

import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


class DynamicsLogger:
    """Logs gradient dynamics to SQLite database for web visualization.

    Tracks simple gradient metrics per expert: norm and variance.

    Usage:
        logger = DynamicsLogger(run_dir="build/runs/83492c812", num_experts=2)
        dynamics = model.router.log_gradient_dynamics()
        logger.log(step=100, dynamics=dynamics)
        logger.close()

    Context manager usage:
        with DynamicsLogger(run_dir="build/runs/83492c812", num_experts=2) as logger:
            dynamics = model.router.log_gradient_dynamics()
            logger.log(step=100, dynamics=dynamics)
    """

    def __init__(
        self, run_dir: str, filename: str = "dynamics.db", num_experts: int = 2
    ):
        """Initialize the dynamics logger.

        Args:
            run_dir: Directory for the current run (e.g., "build/runs/83492c812")
            filename: Name of the database file (default: "dynamics.db")
            num_experts: Number of experts to track (determines schema columns)
        """
        self.run_dir = Path(run_dir)
        self.filepath = self.run_dir / filename
        self.lock = Lock()
        self.num_experts = num_experts

        # Ensure directory exists
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Open database connection
        self.conn = sqlite3.connect(
            self.filepath, check_same_thread=False, timeout=30.0
        )

        # Enable WAL mode for concurrent reads during training
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        # Ensure WAL checkpoints happen more frequently for instant visibility
        # This is critical for gradient dynamics since writes are so infrequent
        self.conn.execute(
            "PRAGMA wal_autocheckpoint=1"
        )  # Checkpoint after every 1 page (~4KB)

        # Create schema
        self._create_schema()

    def _create_schema(self) -> None:
        """Create dynamics table with simple gradient metrics per expert."""
        # Build columns dynamically based on num_experts
        columns = [
            "step INTEGER PRIMARY KEY",
            "ts REAL NOT NULL",
            "num_experts INTEGER",
        ]

        # Simple metrics per expert: gradient norm and variance
        for expert_idx in range(self.num_experts):
            columns.append(f"expert_{expert_idx}_grad_norm REAL")
            columns.append(f"expert_{expert_idx}_grad_var REAL")

        schema = f"""
        CREATE TABLE IF NOT EXISTS dynamics (
            {', '.join(columns)}
        );
        CREATE INDEX IF NOT EXISTS idx_dynamics_ts ON dynamics(ts);
        """

        self.conn.executescript(schema)
        self.conn.commit()

    def _ensure_columns_exist(self, column_names: list) -> None:
        """Ensure all columns exist in the dynamics table, adding missing ones dynamically.

        This handles cases where new metric types are added (e.g., router gradients)
        that weren't in the original schema.

        Args:
            column_names: List of column names to ensure exist
        """
        try:
            # Get existing columns
            cursor = self.conn.cursor()
            cursor.execute("PRAGMA table_info(dynamics)")
            existing_columns = {row[1] for row in cursor.fetchall()}

            # Add missing columns
            for col_name in column_names:
                if col_name not in existing_columns:
                    # SQLite requires ALTER TABLE for each column individually
                    self.conn.execute(
                        f"ALTER TABLE dynamics ADD COLUMN {col_name} REAL"
                    )
                    print(f"[DynamicsLogger] Added new column: {col_name}")

            # Commit schema changes
            self.conn.commit()

        except Exception as e:
            print(f"[DynamicsLogger] Warning: Error ensuring columns exist: {e}")
            # Don't re-raise - continue with existing schema

    def log(self, step: int, dynamics: Optional[Dict[str, Any]]) -> None:
        """Log gradient dynamics for a training step.

        Args:
            step: Training step number
            dynamics: Flat dict from router.log_gradient_dynamics()
                {
                    "expert_0_grad_norm": 0.12,
                    "expert_0_grad_var": 0.003,
                    "expert_1_grad_norm": 0.08,
                    "expert_1_grad_var": 0.002,
                    ...
                }
        """
        if not dynamics:
            return

        try:
            with self.lock:
                # Build column and value lists
                columns = ["step", "ts", "num_experts"]
                values = [step, datetime.now().timestamp(), self.num_experts]

                # Add all dynamics metrics
                for key, value in sorted(dynamics.items()):
                    if not isinstance(value, (int, float)):
                        continue
                    columns.append(key)
                    values.append(value)

                # Ensure all columns exist (handles new experts added dynamically)
                self._ensure_columns_exist(columns[3:])

                # Build UPSERT query
                placeholders = ", ".join(["?"] * len(columns))
                update_clauses = ["ts = excluded.ts", "num_experts = excluded.num_experts"]
                for col in columns[3:]:
                    update_clauses.append(
                        f"{col} = COALESCE(excluded.{col}, dynamics.{col})"
                    )

                query = f"""
                    INSERT INTO dynamics ({', '.join(columns)})
                    VALUES ({placeholders})
                    ON CONFLICT(step) DO UPDATE SET
                    {', '.join(update_clauses)}
                """

                self.conn.execute(query, values)
                self.conn.commit()
                self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

        except Exception as e:
            print(f"[DynamicsLogger] ❌ Error logging dynamics at step {step}: {e}")
            import traceback

            traceback.print_exc()

    def close(self) -> None:
        """Close the database connection."""
        with self.lock:
            if self.conn:
                # Final commit not needed since we commit on every write
                self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass
