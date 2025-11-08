"""Metrics API routes with intelligent downsampling and caching."""

import hashlib
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, current_app, jsonify, request

metrics_bp = Blueprint("metrics", __name__)


def _sanitize_for_json(obj):
    """Recursively sanitize data structure for JSON serialization.

    Converts NaN and Infinity values to None to prevent JSON encoding errors.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


@metrics_bp.route("/api/metrics", methods=["GET", "OPTIONS"])
def get_metrics():
    """Get training metrics with LTTB downsampling and caching.

    Query Parameters:
        since: Only return metrics after this step (default: 0)
        limit: Maximum number of data points to return (default: 1000)
        downsample: Downsampling method - 'lttb' (default: 'lttb')

    Response Headers:
        ETag: Hash of metrics file for caching
        Cache-Control: max-age=5 (cache for 5 seconds)

    Downsampling:
        Uses LTTB (Largest Triangle Three Buckets) algorithm which preserves
        visual fidelity by selecting points that maintain curve shape.

    Returns:
        200: Metrics data
        304: Not Modified (if ETag matches)
        404: No metrics file found
        500: Server error
    """
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add(
            "Access-Control-Allow-Headers", "Content-Type, If-None-Match"
        )
        response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        return response

    try:
        from ..app import api_logger

        # Get query parameters
        since_step = int(request.args.get("since", 0))
        limit = int(request.args.get("limit", 1000))
        downsample_method = request.args.get("downsample", "lttb")
        runs_param = request.args.get("runs", "")  # Comma-separated hashes

        api_logger.debug(
            f"Metrics request: since={since_step}, limit={limit}, downsample={downsample_method}"
        )

        # Get current run directory
        current_hash = current_app.config.get("truncated_hash", "unknown")

        # Parse runs to fetch
        if runs_param:
            run_hashes = [h.strip() for h in runs_param.split(",") if h.strip()]
        else:
            run_hashes = [current_hash]

        # Fetch metrics for each run
        all_runs_data = []
        etag_parts = []

        for run_hash in run_hashes:
            run_dir = Path("build/runs") / run_hash
            metrics_file = run_dir / "metrics.db"

            if not metrics_file.exists():
                continue

            # Calculate ETag component
            stat = metrics_file.stat()
            etag_parts.append(f"{run_hash}:{stat.st_mtime}:{stat.st_size}")

            # Read and parse metrics with SQL-level sampling for efficiency
            # Pass limit * 3 to give LTTB algorithm enough data points to work with
            raw_metrics = _read_metrics_file(
                metrics_file, since_step, max_rows=limit * 3
            )

            if not raw_metrics:
                api_logger.debug(f"No metrics found for run {run_hash}")
                continue

            api_logger.debug(
                f"Loaded {len(raw_metrics)} raw metrics for run {run_hash}"
            )

            # Downsample with LTTB if we still have more points than needed
            if len(raw_metrics) > limit:
                raw_metrics = _downsample_metrics(raw_metrics, limit, downsample_method)
                api_logger.debug(
                    f"Downsampled to {len(raw_metrics)} points for run {run_hash}"
                )

            # Transform to API format
            metrics_data = _transform_metrics(raw_metrics)

            all_runs_data.append(
                {
                    "hash": run_hash,
                    "is_current": run_hash == current_hash,
                    "metrics": metrics_data,
                    "metadata": {
                        "model_hash": run_hash,
                        "last_updated": stat.st_mtime,
                        "num_points": len(raw_metrics),
                        "downsampled": len(raw_metrics) < limit,
                        "first_step": raw_metrics[0]["step"] if raw_metrics else 0,
                        "last_step": raw_metrics[-1]["step"] if raw_metrics else 0,
                    },
                }
            )

        if not all_runs_data:
            return (
                jsonify(
                    {
                        "status": "no_data",
                        "message": "No metrics found for the requested runs",
                    }
                ),
                404,
            )

        # Calculate combined ETag
        etag = hashlib.md5("|".join(etag_parts).encode()).hexdigest()

        # Check if client has cached version
        if request.headers.get("If-None-Match") == etag:
            response = jsonify({"status": "not_modified"})
            response.headers["ETag"] = etag
            return response, 304

        # Build response
        response_data = {
            "status": "ok",
            "source": "metrics_logger",
            "runs": all_runs_data,
            "metadata": {
                "total_params": current_app.config.get("total_params", "N/A"),
                "current_hash": current_hash,
                "num_runs": len(all_runs_data),
            },
        }

        # Sanitize response data to prevent NaN/Infinity JSON encoding errors
        response_data = _sanitize_for_json(response_data)

        # Log response size for debugging
        try:
            import sys

            response_size = sys.getsizeof(json.dumps(response_data))
            api_logger.debug(
                f"Metrics response size: {response_size} bytes ({len(all_runs_data)} runs)"
            )
        except Exception as size_err:
            api_logger.warning(f"Could not calculate response size: {size_err}")

        response = jsonify(response_data)
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = "max-age=5"  # Cache for 5 seconds
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        from ..app import api_logger

        api_logger.error(f"Error in get_metrics endpoint: {e}", exc_info=True)

        error_response = jsonify(
            {"error": str(e), "status": "error", "error_type": type(e).__name__}
        )
        error_response.headers.add("Access-Control-Allow-Origin", "*")
        return error_response, 500


def _read_metrics_file(
    db_path: Path, since_step: int = 0, max_rows: int = None
) -> List[Dict[str, Any]]:
    """Read metrics from SQLite database, filtering by step.

    Args:
        db_path: Path to metrics.db file
        since_step: Only include metrics after this step
        max_rows: Maximum rows to fetch (applies intelligent sampling if needed)

    Returns:
        List of metric dictionaries, sorted by step
    """
    if not db_path.exists():
        return []

    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row  # Access columns by name
        cursor = conn.cursor()

        # If max_rows specified, check total count and sample intelligently
        if max_rows:
            cursor.execute(
                "SELECT COUNT(*) FROM metrics WHERE step >= ?", (since_step,)
            )
            total_count = cursor.fetchone()[0]

            # If dataset is larger than max_rows, use SQL-level sampling
            # Sample every Nth row to reduce memory overhead
            if total_count > max_rows * 2:
                # Use modulo sampling for efficiency - samples evenly across the dataset
                sample_interval = max(1, total_count // max_rows)
                cursor.execute(
                    f"""SELECT step, ts, loss, val_loss, learning_rate, num_tokens,
                              avg_step_time, softmax_collapse, val_perplexity, batch,
                              local_experts, remote_experts, extra_metrics
                       FROM metrics
                       WHERE step >= ? AND (rowid % {sample_interval}) = 0
                       ORDER BY step
                       LIMIT ?""",
                    (since_step, max_rows),
                )
            else:
                # Dataset is small enough, just query normally
                cursor.execute(
                    """SELECT step, ts, loss, val_loss, learning_rate, num_tokens,
                              avg_step_time, softmax_collapse, val_perplexity, batch,
                              local_experts, remote_experts, extra_metrics
                       FROM metrics
                       WHERE step >= ?
                       ORDER BY step""",
                    (since_step,),
                )
        else:
            # No limit specified, query all rows
            cursor.execute(
                """SELECT step, ts, loss, val_loss, learning_rate, num_tokens,
                          avg_step_time, softmax_collapse, val_perplexity, batch,
                          local_experts, remote_experts, extra_metrics
                   FROM metrics
                   WHERE step >= ?
                   ORDER BY step""",
                (since_step,),
            )

        metrics = []
        for row in cursor.fetchall():
            # Build metric dict from native columns
            entry = {
                "step": row["step"],
                "ts": datetime.fromtimestamp(row["ts"]).isoformat(),
            }

            # Add non-null native columns
            for col in [
                "loss",
                "val_loss",
                "learning_rate",
                "num_tokens",
                "avg_step_time",
                "softmax_collapse",
                "val_perplexity",
                "batch",
                "local_experts",
                "remote_experts",
            ]:
                if row[col] is not None:
                    entry[col] = row[col]

            # Merge extra metrics from JSON blob
            if row["extra_metrics"]:
                try:
                    entry.update(json.loads(row["extra_metrics"]))
                except json.JSONDecodeError:
                    pass  # Skip malformed JSON

            metrics.append(entry)

        conn.close()
        return metrics

    except Exception as e:
        from ..app import api_logger

        api_logger.error(f"Error reading metrics from {db_path}: {e}", exc_info=True)
        return []


def _downsample_metrics(
    metrics: List[Dict[str, Any]], target_size: int, method: str = "lttb"
) -> List[Dict[str, Any]]:
    """Downsample metrics using LTTB (Largest Triangle Three Buckets).

    LTTB preserves visual fidelity by selecting points that maintain the shape
    of the time-series curve, avoiding the temporal distortion of index-based sampling.

    Args:
        metrics: List of metric dictionaries
        target_size: Target number of points
        method: Downsampling method (only 'lttb' supported now)

    Returns:
        Downsampled list of metrics using LTTB algorithm
    """
    if len(metrics) <= target_size:
        return metrics

    # LTTB Algorithm (Largest Triangle Three Buckets)
    # Select points that maximize the area of triangles formed with neighbors
    # This preserves visual shape and trends better than uniform sampling

    if target_size < 3:
        # For very small target sizes, just return first, middle, last
        if target_size == 1:
            return [metrics[-1]]
        return [metrics[0], metrics[-1]]

    # Store selected indices instead of points directly
    selected_indices = [0]  # Always include first point

    # Bucket size (average number of points per bucket)
    bucket_size = (len(metrics) - 2) / (target_size - 2)

    for bucket_idx in range(target_size - 2):
        # Calculate bucket range
        bucket_start = int((bucket_idx + 0) * bucket_size) + 1
        bucket_end = int((bucket_idx + 1) * bucket_size) + 1

        # Calculate the next bucket's average point (for triangle calculation)
        next_bucket_start = int((bucket_idx + 1) * bucket_size) + 1
        next_bucket_end = min(int((bucket_idx + 2) * bucket_size) + 1, len(metrics))

        # Calculate average point in next bucket
        next_avg_x = 0
        next_avg_y = 0
        next_count = 0

        for i in range(next_bucket_start, next_bucket_end):
            if i >= len(metrics):
                break
            next_avg_x += metrics[i].get("step", i)
            # Use 'loss' as primary metric for area calculation
            next_avg_y += metrics[i].get("loss", metrics[i].get("val_loss", 0))
            next_count += 1

        if next_count > 0:
            next_avg_x /= next_count
            next_avg_y /= next_count

        # Find point in current bucket that maximizes triangle area
        prev_idx = selected_indices[-1]
        prev_point = metrics[prev_idx]
        prev_x = prev_point.get("step", prev_idx)
        prev_y = prev_point.get("loss", prev_point.get("val_loss", 0))

        max_area = -1
        max_area_point = None

        for i in range(bucket_start, bucket_end):
            if i >= len(metrics):
                break

            curr_x = metrics[i].get("step", i)
            curr_y = metrics[i].get("loss", metrics[i].get("val_loss", 0))

            # Calculate triangle area
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

    selected_indices.append(len(metrics) - 1)  # Always include last point

    # Sort indices to maintain chronological order, then extract points
    selected_indices.sort()
    downsampled = [metrics[i] for i in selected_indices]

    # CRITICAL: Sort by actual step values to ensure monotonic time progression
    # This prevents Chart.js from drawing backwards/zigzag lines
    downsampled.sort(key=lambda m: m.get("step", 0))

    # CRITICAL: Deduplicate by step - keep only the last occurrence of each step
    # This prevents vertical lines when multiple points share the same step value
    seen_steps = {}
    for metric in downsampled:
        step = metric.get("step", 0)
        seen_steps[step] = metric  # Overwrites earlier occurrences

    # Return deduplicated list, sorted by step
    deduplicated = list(seen_steps.values())
    deduplicated.sort(key=lambda m: m.get("step", 0))

    return deduplicated


def _transform_metrics(raw_metrics: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Transform list of metric dicts to column format for charts.

    Input: [{"step": 0, "loss": 2.45}, {"step": 100, "loss": 2.12, "val_loss": 2.20}]
    Output: {"steps": [0, 100], "loss": [2.45, 2.12], "val_loss": [null, 2.20]}

    Args:
        raw_metrics: List of metric dictionaries

    Returns:
        Dictionary with metric names as keys and lists as values
    """
    if not raw_metrics:
        return {}

    # Collect all unique keys
    all_keys = set()
    for entry in raw_metrics:
        all_keys.update(entry.keys())

    # Remove metadata keys
    all_keys.discard("ts")

    # Initialize result dict
    result = {key: [] for key in all_keys}

    # Transform to column format
    for entry in raw_metrics:
        for key in all_keys:
            result[key].append(entry.get(key, None))

    # Rename 'step' to 'steps' for consistency
    if "step" in result:
        result["steps"] = result.pop("step")

    return result


@metrics_bp.route("/api/runs", methods=["GET", "OPTIONS"])
def get_runs():
    """Get list of available training runs.

    Returns:
        200: List of runs with metadata
        500: Server error
    """
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        return response

    try:
        runs_dir = Path("build/runs")

        if not runs_dir.exists():
            return jsonify({"status": "ok", "runs": []})

        runs = []
        current_hash = current_app.config.get("truncated_hash", "unknown")

        # Scan all run directories
        for run_path in runs_dir.iterdir():
            if not run_path.is_dir():
                continue

            truncated_hash = run_path.name
            config_file = run_path / "config.json"
            metrics_file = run_path / "metrics.db"

            # Skip if no metrics file
            if not metrics_file.exists():
                continue

            # Load config if available
            config = {}
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        config = json.load(f)
                except:
                    pass

            # Get metrics file stats
            stat = metrics_file.stat()

            # Count rows in SQLite database to get step count
            num_steps = 0
            try:
                conn = sqlite3.connect(metrics_file, timeout=30.0)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM metrics")
                num_steps = cursor.fetchone()[0]
                conn.close()
            except:
                pass

            runs.append(
                {
                    "hash": truncated_hash,
                    "is_current": truncated_hash == current_hash,
                    "command": config.get("command", "unknown"),
                    "created": config.get("created"),
                    "last_updated": config.get("last_updated"),
                    "metrics_updated": stat.st_mtime,
                    "num_steps": num_steps,
                    "has_metrics": metrics_file.exists(),
                }
            )

        # Sort by last updated (most recent first)
        runs.sort(key=lambda x: x.get("metrics_updated", 0), reverse=True)

        response_data = {"status": "ok", "runs": runs, "current_hash": current_hash}

        response = jsonify(response_data)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        from ..app import api_logger

        api_logger.error(f"Error in get_runs endpoint: {e}", exc_info=True)

        error_response = jsonify(
            {"error": str(e), "status": "error", "error_type": type(e).__name__}
        )
        error_response.headers.add("Access-Control-Allow-Origin", "*")
        return error_response, 500
