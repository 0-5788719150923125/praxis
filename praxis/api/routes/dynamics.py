"""Gradient dynamics API routes for Expert learning visualization."""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, current_app, jsonify, request

from praxis.routers.prismatic import get_pi_digit_at

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

        # Detect number of experts and compute pi-phase metadata
        expert_metadata = _compute_expert_metadata(dynamics_data["metrics"])

        # Format response
        response_data = {
            "status": "ok",
            "runs": [{
                "hash": current_hash,
                "metadata": {
                    "num_points": dynamics_data["num_points"],
                    "last_step": dynamics_data.get("last_step", 0),
                    **expert_metadata  # Add expert count and pi-seeds
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
    Read gradient dynamics from SQLite database and merge with routing weights.

    Schema:
        dynamics.db:
            step INTEGER PRIMARY KEY,
            expert_0_top_norm REAL,
            expert_0_bottom_norm REAL,
            expert_0_middle_norm REAL,
            expert_1_top_norm REAL,
            expert_1_bottom_norm REAL,
            expert_1_middle_norm REAL,
            expert_1_divergence REAL,
            ...

        metrics.db:
            step INTEGER PRIMARY KEY,
            extra_metrics TEXT (JSON with routing_weight, routing_entropy, etc.)

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

        # Merge routing weights from metrics.db
        metrics_db_path = db_path.parent / "metrics.db"
        if metrics_db_path.exists():
            routing_data = _read_routing_weights_from_metrics(
                metrics_db_path, metrics['step']
            )
            if routing_data:
                metrics.update(routing_data)

        return {
            "metrics": metrics,
            "num_points": len(rows),
            "last_step": rows[-1][0] if rows else 0
        }

    except Exception as e:
        print(f"Error reading dynamics db: {e}")
        return None


def _read_routing_weights_from_metrics(
    metrics_db_path: Path, steps: List[int]
) -> Optional[Dict[str, List]]:
    """
    Read routing weights from metrics.db extra_metrics JSON field.

    Handles cases where not all dynamics steps have corresponding routing metrics.
    Missing values are filled with None or interpolated from nearest neighbors.

    Args:
        metrics_db_path: Path to metrics.db
        steps: List of steps to query (from dynamics.db)

    Returns:
        Dict with routing weight arrays keyed by expert_i_routing_weight
    """
    try:
        import json

        conn = sqlite3.connect(f'file:{metrics_db_path}?mode=ro', uri=True)
        cursor = conn.cursor()

        # Query ALL routing metrics in the step range (not just exact matches)
        min_step = min(steps)
        max_step = max(steps)

        query = f"""
            SELECT step, extra_metrics
            FROM metrics
            WHERE step >= ? AND step <= ? AND extra_metrics IS NOT NULL
            ORDER BY step ASC
        """

        cursor.execute(query, (min_step, max_step))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        # Parse routing weights from JSON and create step-indexed map
        routing_data_by_step = {}

        for step, extra_metrics_json in rows:
            try:
                extra_metrics = json.loads(extra_metrics_json)
                routing_data_by_step[step] = {
                    k: v for k, v in extra_metrics.items() if 'routing' in k
                }
            except json.JSONDecodeError:
                continue

        if not routing_data_by_step:
            return None

        # Build aligned arrays for the requested steps
        # First, determine all routing metric keys from all available data
        all_routing_keys = set()
        for step_data in routing_data_by_step.values():
            all_routing_keys.update(step_data.keys())

        if not all_routing_keys:
            return None

        # Initialize arrays for all routing metrics
        routing_metrics = {key: [None] * len(steps) for key in all_routing_keys}

        # Fill in values using forward-fill for missing steps
        last_known_values = {}

        for step_idx, step in enumerate(steps):
            # Try exact match first
            if step in routing_data_by_step:
                step_data = routing_data_by_step[step]
                # Update last known values
                last_known_values.update(step_data)
            else:
                # Use last known values (forward-fill)
                step_data = last_known_values

            # Fill in this step's data
            for key in all_routing_keys:
                if key in step_data:
                    routing_metrics[key][step_idx] = step_data[key]

        return routing_metrics if routing_metrics else None

    except Exception as e:
        print(f"Error reading routing weights from metrics: {e}")
        import traceback
        traceback.print_exc()
        return None


def _compute_expert_metadata(metrics: Dict[str, List]) -> Dict[str, Any]:
    """
    Compute expert metadata including pi-digit seeds for Quantum Echoes visualization.

    Detects number of experts from dynamics data and computes the pi digit
    each expert was seeded with (walking backwards from pi_position=100000).

    Args:
        metrics: Dynamics metrics dict with keys like "expert_0_top_norm", etc.

    Returns:
        Dict with:
            - num_experts: int
            - pi_seeds: List[int] - pi digit for each expert (0-9)
            - pi_phases: List[float] - phase angle in radians (0 to 2π)
    """
    # Detect number of experts from metric keys
    expert_keys = [k for k in metrics.keys() if k.startswith("expert_")]
    if not expert_keys:
        return {"num_experts": 0, "pi_seeds": [], "pi_phases": []}

    # Extract expert indices
    expert_indices = set()
    for key in expert_keys:
        parts = key.split("_")
        if len(parts) >= 2 and parts[0] == "expert":
            try:
                expert_indices.add(int(parts[1]))
            except ValueError:
                continue

    num_experts = len(expert_indices)
    if num_experts == 0:
        return {"num_experts": 0, "pi_seeds": [], "pi_phases": []}

    # Compute pi seeds for each expert (Quantum Echoes)
    # Expert 0: no seed (clean)
    # Expert 1: pi[position - 1], Expert 2: pi[position - 2], etc.
    pi_seeds = []
    pi_phases = []
    pi_position = 100000  # Default position (matches Prismatic default)

    for expert_idx in sorted(expert_indices):
        if expert_idx == 0:
            # Expert 0 is clean (no pi seed)
            pi_seeds.append(None)
            pi_phases.append(0.0)
        else:
            # Walk backwards through pi
            pi_index = pi_position - expert_idx
            pi_digit = get_pi_digit_at(pi_index)
            pi_seeds.append(pi_digit)
            # Map digit (0-9) to phase angle (0 to 2π)
            phase = (pi_digit / 10.0) * 2 * 3.141592653589793
            pi_phases.append(phase)

    return {
        "num_experts": num_experts,
        "pi_seeds": pi_seeds,
        "pi_phases": pi_phases
    }
