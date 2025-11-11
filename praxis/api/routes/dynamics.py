"""Gradient dynamics API routes for Expert learning visualization."""

import math
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
            response = jsonify(
                {
                    "status": "no_data",
                    "message": "Gradient dynamics not logged. Enable by calling router.log_gradient_dynamics() during training.",
                    "runs": [],
                }
            )
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response

        api_logger.info(
            f"Found dynamics file: {dynamics_file}, size: {dynamics_file.stat().st_size} bytes"
        )

        # Read dynamics from SQLite database
        # Pass limit * 3 to give LTTB algorithm enough data points to work with
        try:
            dynamics_data = _read_dynamics_from_db(dynamics_file, since_step, limit * 3)
        except Exception as read_error:
            api_logger.error(f"Error reading dynamics database: {read_error}")
            import traceback

            traceback.print_exc()
            response = jsonify(
                {
                    "status": "error",
                    "message": f"Error reading database: {str(read_error)}",
                    "runs": [],
                }
            )
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.status_code = 500
            return response

        if not dynamics_data or dynamics_data.get("num_points", 0) == 0:
            api_logger.warning(f"No data points found in dynamics database")
            response = jsonify(
                {
                    "status": "no_data",
                    "message": "No dynamics data points found",
                    "runs": [],
                }
            )
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response

        # Apply LTTB downsampling if we have more points than requested
        original_count = dynamics_data["num_points"]
        if original_count > limit:
            dynamics_data["metrics"] = _downsample_dynamics_lttb(
                dynamics_data["metrics"], limit
            )
            dynamics_data["num_points"] = len(dynamics_data["metrics"]["steps"])
            api_logger.debug(
                f"Downsampled dynamics from {original_count} to {dynamics_data['num_points']} points"
            )

        # Detect number of experts and compute metadata for charts
        # First check if num_experts is stored in the database
        stored_num_experts = dynamics_data.get("num_experts")
        expert_metadata = _compute_expert_metadata(
            dynamics_data["metrics"], dynamics_data.get("metadata"), stored_num_experts
        )

        # Format response
        response_data = {
            "status": "ok",
            "runs": [
                {
                    "hash": current_hash,
                    "metadata": {
                        "num_points": dynamics_data["num_points"],
                        "last_step": dynamics_data.get("last_step", 0),
                        **expert_metadata,  # Add expert count, pi_phases, pi_seeds
                    },
                    "dynamics": dynamics_data["metrics"],
                }
            ],
        }

        response = jsonify(response_data)
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Cache-Control", "max-age=5")

        return response

    except Exception as e:
        from ..app import api_logger

        api_logger.error(f"Error in get_dynamics: {str(e)}")

        response = jsonify({"status": "error", "message": str(e), "runs": []})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.status_code = 500
        return response


def _downsample_dynamics_lttb(
    metrics: Dict[str, List], target_size: int
) -> Dict[str, List]:
    """Downsample dynamics metrics using LTTB algorithm.

    LTTB (Largest Triangle Three Buckets) preserves visual fidelity by selecting
    points that maintain the shape of time-series curves.

    Args:
        metrics: Dict with "steps" and metric arrays (e.g., "expert_0_grad_norm")
        target_size: Target number of points

    Returns:
        Downsampled metrics dict with same structure
    """
    steps = metrics.get("steps") or metrics.get("step", [])
    if len(steps) <= target_size:
        return metrics

    if target_size < 3:
        # For very small targets, return first and last
        if target_size == 1:
            indices = [len(steps) - 1]
        else:
            indices = [0, len(steps) - 1]
    else:
        # LTTB algorithm
        selected_indices = [0]  # Always include first point

        bucket_size = (len(steps) - 2) / (target_size - 2)

        # For each bucket
        for bucket_idx in range(target_size - 2):
            # Current bucket range
            curr_bucket_start = int(bucket_idx * bucket_size) + 1
            curr_bucket_end = int((bucket_idx + 1) * bucket_size) + 1

            # Previous point
            prev_idx = selected_indices[-1]
            prev_x = steps[prev_idx]

            # Next bucket average (for triangle calculation)
            next_bucket_start = curr_bucket_end
            next_bucket_end = min(int((bucket_idx + 2) * bucket_size) + 1, len(steps))

            if next_bucket_end > len(steps):
                next_bucket_end = len(steps)

            if next_bucket_start >= len(steps):
                break

            next_avg_x = sum(steps[next_bucket_start:next_bucket_end]) / (
                next_bucket_end - next_bucket_start
            )

            # Use first metric array for Y values (typically a gradient norm)
            # Find first non-step metric
            y_key = None
            for key in metrics.keys():
                if key not in ("steps", "step"):
                    y_key = key
                    break

            if not y_key:
                # No metric data, just sample uniformly
                selected_indices.extend(
                    range(
                        curr_bucket_start,
                        min(curr_bucket_end, len(steps)),
                        max(1, (curr_bucket_end - curr_bucket_start) // (target_size - len(selected_indices))),
                    )
                )
                continue

            y_values = metrics[y_key]
            prev_y = y_values[prev_idx]

            # Calculate next bucket avg y
            next_avg_y = sum(y_values[next_bucket_start:next_bucket_end]) / (
                next_bucket_end - next_bucket_start
            )

            # Find point in current bucket that maximizes triangle area
            max_area = -1
            max_area_point = None

            for i in range(curr_bucket_start, min(curr_bucket_end, len(steps))):
                curr_x = steps[i]
                curr_y = y_values[i]

                # Triangle area formula
                area = (
                    abs(
                        (prev_x - next_avg_x) * (curr_y - prev_y)
                        - (prev_x - curr_x) * (next_avg_y - prev_y)
                    )
                    * 0.5
                )

                if area > max_area:
                    max_area = area
                    max_area_point = i

            if max_area_point is not None:
                selected_indices.append(max_area_point)

        selected_indices.append(len(steps) - 1)  # Always include last point
        indices = sorted(list(set(selected_indices)))  # Deduplicate and sort

    # Apply indices to all metric arrays
    downsampled = {}
    for key, values in metrics.items():
        if isinstance(values, list) and len(values) == len(steps):
            downsampled[key] = [values[i] for i in indices]
        else:
            downsampled[key] = values

    return downsampled


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
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
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
        if "step" in metrics and "steps" not in metrics:
            metrics["steps"] = metrics["step"]

        # Merge routing weights from metrics.db
        metrics_db_path = db_path.parent / "metrics.db"
        if metrics_db_path.exists():
            routing_data = _read_routing_weights_from_metrics(
                metrics_db_path, metrics["step"]
            )
            if routing_data:
                metrics.update(routing_data)

        # Extract num_experts from the data if available
        num_experts = None
        if "num_experts" in metrics and len(metrics["num_experts"]) > 0:
            # Get the most recent num_experts value
            num_experts = metrics["num_experts"][-1]

        return {
            "metrics": metrics,
            "num_points": len(rows),
            "last_step": rows[-1][0] if rows else 0,
            "num_experts": num_experts,
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

        conn = sqlite3.connect(f"file:{metrics_db_path}?mode=ro", uri=True)
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
                # Extract routing metrics and weight divergence metrics
                routing_data_by_step[step] = {
                    k: v
                    for k, v in extra_metrics.items()
                    if "routing" in k or "cosine" in k or "weight_angle" in k
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


def _compute_expert_metadata(
    metrics: Dict[str, List],
    extra_metadata: Optional[Dict] = None,
    stored_num_experts: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute expert metadata from dynamics data.

    Args:
        metrics: Dynamics metrics dict
        extra_metadata: Optional additional metadata
        stored_num_experts: Number of experts from database

    Returns:
        Dict with num_experts
    """
    # Use stored num_experts if available
    if stored_num_experts is not None:
        return {"num_experts": int(stored_num_experts)}

    # Detect from grad_norm metrics
    expert_keys = [k for k in metrics.keys() if k.startswith("expert_") and "_grad_norm" in k]

    if not expert_keys:
        return {"num_experts": 0}

    # Extract expert indices
    expert_indices = set()
    for key in expert_keys:
        parts = key.split("_")
        if len(parts) >= 2 and parts[0] == "expert":
            try:
                expert_indices.add(int(parts[1]))
            except ValueError:
                continue

    return {"num_experts": len(expert_indices)}
