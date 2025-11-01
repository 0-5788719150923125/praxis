"""Gradient dynamics API routes for Expert learning visualization."""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, current_app, jsonify, request

dynamics_bp = Blueprint("dynamics", __name__)


@dynamics_bp.route("/api/dynamics", methods=["GET", "OPTIONS"])
def get_dynamics():
    """Get gradient dynamics data for expert learning visualization.

    Query Parameters:
        since: Only return dynamics after this step (default: 0)
        limit: Maximum number of data points to return (default: 1000)

    Returns:
        200: Dynamics data with expert gradients by tier
        404: No dynamics data found
        500: Server error

    Data Structure:
        {
            "status": "ok",
            "runs": [{
                "hash": "abc123",
                "metadata": { "num_points": 100 },
                "dynamics": {
                    "steps": [0, 100, 200, ...],
                    "expert_0_top_norm": [0.12, 0.11, ...],
                    "expert_0_bottom_norm": [0.003, 0.004, ...],
                    "expert_0_middle_norm": [0.05, 0.04, ...],
                    "expert_1_top_norm": [0.08, 0.09, ...],
                    "expert_1_bottom_norm": [0.012, 0.015, ...],  # Awakened?
                    "expert_1_middle_norm": [0.04, 0.04, ...],
                    "expert_1_divergence": [0.05, 0.06, ...]
                }
            }]
        }

    Note:
        Gradient logging must be enabled in training loop by calling:
        `router.log_gradient_dynamics()` after backward() but before step().

        See docs/gradient_visualization_proposals.md for integration details.
    """
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        return response

    try:
        from ..app import api_logger

        # Get query parameters
        since_step = int(request.args.get("since", 0))
        limit = int(request.args.get("limit", 1000))

        api_logger.debug(f"Dynamics request: since={since_step}, limit={limit}")

        # Get current run directory
        current_hash = current_app.config.get("truncated_hash", "unknown")
        run_dir = Path("build/runs") / current_hash
        dynamics_file = run_dir / "dynamics.db"

        # Check if dynamics data exists
        if not dynamics_file.exists():
            # Return empty state with helpful message
            api_logger.warning(f"Dynamics file not found: {dynamics_file}")
            response = jsonify({
                "status": "no_data",
                "message": "Gradient dynamics not logged. Enable by calling router.log_gradient_dynamics() during training.",
                "runs": []
            })
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response

        api_logger.info(f"Found dynamics file: {dynamics_file}, size: {dynamics_file.stat().st_size} bytes")

        # Read dynamics from SQLite database
        try:
            dynamics_data = _read_dynamics_from_db(
                dynamics_file, since_step, limit
            )
        except Exception as read_error:
            api_logger.error(f"Error reading dynamics database: {read_error}")
            import traceback
            traceback.print_exc()
            response = jsonify({
                "status": "error",
                "message": f"Error reading database: {str(read_error)}",
                "runs": []
            })
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.status_code = 500
            return response

        if not dynamics_data or dynamics_data.get("num_points", 0) == 0:
            api_logger.warning(f"No data points found in dynamics database")
            response = jsonify({
                "status": "no_data",
                "message": "No dynamics data points found",
                "runs": []
            })
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response

        # Format response
        response_data = {
            "status": "ok",
            "runs": [{
                "hash": current_hash,
                "metadata": {
                    "num_points": dynamics_data["num_points"],
                    "last_step": dynamics_data.get("last_step", 0)
                },
                "dynamics": dynamics_data["metrics"]
            }]
        }

        response = jsonify(response_data)
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Cache-Control", "max-age=5")

        return response

    except Exception as e:
        from ..app import api_logger
        api_logger.error(f"Error in get_dynamics: {str(e)}")

        response = jsonify({
            "status": "error",
            "message": str(e),
            "runs": []
        })
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.status_code = 500
        return response


def _read_dynamics_from_db(
    db_path: Path, since_step: int = 0, limit: int = 1000
) -> Optional[Dict[str, Any]]:
    """
    Read gradient dynamics from SQLite database.

    Schema:
        CREATE TABLE dynamics (
            step INTEGER PRIMARY KEY,
            expert_0_top_norm REAL,
            expert_0_bottom_norm REAL,
            expert_0_middle_norm REAL,
            expert_1_top_norm REAL,
            expert_1_bottom_norm REAL,
            expert_1_middle_norm REAL,
            expert_1_divergence REAL,
            ...
        )

    Args:
        db_path: Path to dynamics SQLite database
        since_step: Only return steps >= this value
        limit: Maximum number of rows to return

    Returns:
        Dict with dynamics arrays or None if no data
    """
    try:
        # Open in read-only mode to avoid permission issues
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        cursor = conn.cursor()

        # Get all column names
        cursor.execute("PRAGMA table_info(dynamics)")
        columns = [row[1] for row in cursor.fetchall()]

        if not columns:
            conn.close()
            return None

        # Query dynamics data
        query = f"""
            SELECT * FROM dynamics
            WHERE step >= ?
            ORDER BY step ASC
            LIMIT ?
        """

        cursor.execute(query, (since_step, limit))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        # Convert to columnar format
        metrics = {col: [] for col in columns}

        for row in rows:
            for i, col in enumerate(columns):
                metrics[col].append(row[i])

        # Ensure 'steps' (plural) exists for frontend compatibility
        if 'step' in metrics and 'steps' not in metrics:
            metrics['steps'] = metrics['step']

        return {
            "metrics": metrics,
            "num_points": len(rows),
            "last_step": rows[-1][0] if rows else 0
        }

    except Exception as e:
        print(f"Error reading dynamics db: {e}")
        return None
