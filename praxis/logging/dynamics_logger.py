"""SQLite-based dynamics logger for gradient visualization."""

import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


class DynamicsLogger:
    """Logs gradient dynamics to SQLite database for web visualization.

    Tracks expert-level gradient norms by weight tier (top/bottom/middle) to validate
    whether dual-sided perturbations force genuinely different learning dynamics.

    Usage:
        logger = DynamicsLogger(run_dir="build/runs/83492c812")
        dynamics = model.router.log_gradient_dynamics()
        logger.log(step=100, dynamics=dynamics)
        logger.close()

    Context manager usage:
        with DynamicsLogger(run_dir="build/runs/83492c812") as logger:
            dynamics = model.router.log_gradient_dynamics()
            logger.log(step=100, dynamics=dynamics)
    """

    def __init__(self, run_dir: str, filename: str = "dynamics.db", num_experts: int = 2):
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
        self.conn.execute("PRAGMA wal_autocheckpoint=1")  # Checkpoint after every 1 page (~4KB)

        # Create schema
        self._create_schema()

    def _create_schema(self) -> None:
        """Create dynamics table with columns for all experts and tiers."""
        # Build columns dynamically based on num_experts
        columns = ["step INTEGER PRIMARY KEY", "ts REAL NOT NULL"]

        tiers = ['top', 'bottom', 'middle', 'clean', 'perturbed']
        metrics = ['norm', 'max', 'min']

        for expert_idx in range(self.num_experts):
            for tier in tiers:
                for metric in metrics:
                    col_name = f"expert_{expert_idx}_{tier}_{metric}"
                    columns.append(f"{col_name} REAL")

            # Divergence scores (for perturbed experts)
            if expert_idx > 0:
                columns.append(f"expert_{expert_idx}_divergence REAL")

        schema = f"""
        CREATE TABLE IF NOT EXISTS dynamics (
            {', '.join(columns)}
        );
        CREATE INDEX IF NOT EXISTS idx_dynamics_ts ON dynamics(ts);
        """

        self.conn.executescript(schema)
        self.conn.commit()

    def log(self, step: int, dynamics: Optional[Dict[str, Any]]) -> None:
        """Log gradient dynamics for a training step.

        Args:
            step: Training step number
            dynamics: Dynamics dict from router.log_gradient_dynamics()
                {
                    'expert_gradients': {
                        'expert_0': {'top_norm': 0.12, 'bottom_norm': 0.003, ...},
                        'expert_1': {'top_norm': 0.08, 'bottom_norm': 0.012, ...}
                    },
                    'divergence_scores': {
                        'expert_1_divergence': 0.05
                    }
                }
        """
        if not dynamics:
            return

        try:
            with self.lock:
                # Build column and value lists
                columns = ["step", "ts"]
                values = [step, datetime.now().timestamp()]

                # Extract expert gradients
                expert_grads = dynamics.get('expert_gradients', {})

                for expert_key in sorted(expert_grads.keys()):
                    expert_data = expert_grads[expert_key]
                    for metric_key, value in sorted(expert_data.items()):
                        col_name = f"{expert_key}_{metric_key}"
                        columns.append(col_name)
                        values.append(value if value is not None else None)

                # Extract divergence scores
                divergence_scores = dynamics.get('divergence_scores', {})
                for div_key, value in sorted(divergence_scores.items()):
                    columns.append(div_key)
                    values.append(value if value is not None else None)

                # Build UPSERT query
                placeholders = ', '.join(['?'] * len(columns))

                # Create update clauses (keep latest non-null values)
                update_clauses = ["ts = excluded.ts"]
                for col in columns[2:]:  # Skip step and ts
                    update_clauses.append(f"{col} = COALESCE(excluded.{col}, dynamics.{col})")

                query = f"""
                    INSERT INTO dynamics ({', '.join(columns)})
                    VALUES ({placeholders})
                    ON CONFLICT(step) DO UPDATE SET
                    {', '.join(update_clauses)}
                """

                self.conn.execute(query, values)

                # Gradient dynamics are so infrequent (every 10 training steps by default)
                # that we should just commit immediately every time for instant dashboard visibility
                self.conn.commit()

                # Force WAL checkpoint to ensure data is immediately visible to readers
                # This is critical since the dashboard polls for new data frequently
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
