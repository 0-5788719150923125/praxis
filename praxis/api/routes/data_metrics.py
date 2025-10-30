"""Data metrics API routes for sampling weights and preprocessing metrics."""

import hashlib
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, current_app, jsonify, request

data_metrics_bp = Blueprint("data_metrics", __name__)


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


@data_metrics_bp.route("/api/data-metrics", methods=["GET", "OPTIONS"])
def get_data_metrics():
    """Get data preprocessing metrics (sampling weights, etc.) with downsampling and caching.

    Query Parameters:
        since: Only return metrics after this step (default: 0)
        limit: Maximum number of data points to return (default: 1000)
        downsample: Downsampling method - 'lttb' (default: 'lttb')
        runs: Comma-separated run hashes to fetch (default: current run)

    Response Headers:
        ETag: Hash of metrics file for caching
        Cache-Control: max-age=5 (cache for 5 seconds)

    Returns:
        200: Data metrics
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

        api_logger.debug(f"Data metrics request: since={since_step}, limit={limit}")

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
            data_metrics_file = run_dir / "data_metrics.db"

            if not data_metrics_file.exists():
                continue

            # Calculate ETag component
            stat = data_metrics_file.stat()
            etag_parts.append(f"{run_hash}:{stat.st_mtime}:{stat.st_size}")

            # Read and parse data metrics with SQL-level sampling
            raw_metrics = _read_data_metrics_file(data_metrics_file, since_step, max_rows=limit * 2)

            if not raw_metrics:
                api_logger.debug(f"No data metrics found for run {run_hash}")
                continue

            api_logger.debug(f"Loaded {len(raw_metrics)} data metrics for run {run_hash}")

            # Downsample if needed
            if len(raw_metrics) > limit and downsample_method == "lttb":
                raw_metrics = _downsample_data_metrics(raw_metrics, limit)
                api_logger.debug(f"Downsampled to {len(raw_metrics)} data points for run {run_hash}")

            # Transform to API format (column-based)
            metrics_data = _transform_data_metrics(raw_metrics)

            all_runs_data.append(
                {
                    "hash": run_hash,
                    "is_current": run_hash == current_hash,
                    "data_metrics": metrics_data,
                    "metadata": {
                        "model_hash": run_hash,
                        "last_updated": stat.st_mtime,
                        "num_points": len(raw_metrics),
                        "downsampled": len(raw_metrics) > limit,
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
                        "message": "No data metrics found for the requested runs",
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
            "source": "data_metrics_logger",
            "runs": all_runs_data,
            "metadata": {"current_hash": current_hash, "num_runs": len(all_runs_data)},
        }

        # Sanitize response data to prevent NaN/Infinity JSON encoding errors
        response_data = _sanitize_for_json(response_data)

        response = jsonify(response_data)
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = "max-age=5"  # Cache for 5 seconds
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        from ..app import api_logger
        api_logger.error(f"Error in get_data_metrics endpoint: {e}", exc_info=True)

        error_response = jsonify({
            "error": str(e),
            "status": "error",
            "error_type": type(e).__name__
        })
        error_response.headers.add("Access-Control-Allow-Origin", "*")
        return error_response, 500


def _read_data_metrics_file(
    db_path: Path, since_step: int = 0, max_rows: int = None
) -> List[Dict[str, Any]]:
    """Read data metrics from SQLite database, filtering by step.

    Args:
        db_path: Path to data_metrics.db file
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
            cursor.execute("SELECT COUNT(*) FROM data_metrics WHERE step >= ?", (since_step,))
            total_count = cursor.fetchone()[0]

            # If dataset is larger than max_rows, use SQL-level sampling
            if total_count > max_rows * 2:
                sample_interval = max(1, total_count // max_rows)
                cursor.execute(
                    f"""SELECT step, ts, sampling_weights, dataset_stats, extra_metrics
                       FROM data_metrics
                       WHERE step >= ? AND (rowid % {sample_interval}) = 0
                       ORDER BY step
                       LIMIT ?""",
                    (since_step, max_rows),
                )
            else:
                cursor.execute(
                    """SELECT step, ts, sampling_weights, dataset_stats, extra_metrics
                       FROM data_metrics
                       WHERE step >= ?
                       ORDER BY step""",
                    (since_step,),
                )
        else:
            # No limit specified, query all rows
            cursor.execute(
                """SELECT step, ts, sampling_weights, dataset_stats, extra_metrics
                   FROM data_metrics
                   WHERE step >= ?
                   ORDER BY step""",
                (since_step,),
            )

        metrics = []
        for row in cursor.fetchall():
            # Build metric dict
            entry = {"step": row["step"], "ts": datetime.fromtimestamp(row["ts"]).isoformat()}

            # Parse JSON fields
            if row["sampling_weights"]:
                try:
                    entry["sampling_weights"] = json.loads(row["sampling_weights"])
                except json.JSONDecodeError:
                    pass

            if row["dataset_stats"]:
                try:
                    entry["dataset_stats"] = json.loads(row["dataset_stats"])
                except json.JSONDecodeError:
                    pass

            if row["extra_metrics"]:
                try:
                    entry.update(json.loads(row["extra_metrics"]))
                except json.JSONDecodeError:
                    pass

            metrics.append(entry)

        conn.close()
        return metrics

    except Exception as e:
        from ..app import api_logger
        api_logger.error(f"Error reading data metrics from {db_path}: {e}", exc_info=True)
        return []


def _downsample_data_metrics(
    metrics: List[Dict[str, Any]], target_size: int
) -> List[Dict[str, Any]]:
    """Downsample data metrics using simple uniform sampling.

    For data metrics (like sampling weights), we use simpler downsampling
    since the values tend to change more smoothly than training loss.

    Args:
        metrics: List of metric dictionaries
        target_size: Target number of points

    Returns:
        Downsampled list of metrics
    """
    if len(metrics) <= target_size:
        return metrics

    # Always include first and last points
    if target_size < 3:
        if target_size == 1:
            return [metrics[-1]]
        return [metrics[0], metrics[-1]]

    # Calculate step size for uniform sampling
    step_size = (len(metrics) - 1) / (target_size - 1)

    selected = [metrics[0]]  # Always include first

    for i in range(1, target_size - 1):
        idx = int(i * step_size)
        selected.append(metrics[idx])

    selected.append(metrics[-1])  # Always include last

    return selected


def _transform_data_metrics(raw_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Transform raw metrics to column-based format for charts.

    Input format (row-based):
        [{"step": 0, "ts": "...", "sampling_weights": {"doc1": 0.5, "doc2": 0.5}}, ...]

    Output format (column-based):
        {
            "steps": [0, 50, 100, ...],
            "timestamps": ["...", "...", ...],
            "sampling_weights": {
                "doc1": [0.5, 0.4, 0.3, ...],
                "doc2": [0.5, 0.6, 0.7, ...]
            }
        }

    Args:
        raw_metrics: List of raw metric dictionaries

    Returns:
        Column-based metrics dictionary
    """
    if not raw_metrics:
        return {"steps": [], "timestamps": [], "sampling_weights": {}}

    steps = []
    timestamps = []

    # Collect all dataset names across all samples
    all_dataset_names = set()
    for entry in raw_metrics:
        if "sampling_weights" in entry:
            all_dataset_names.update(entry["sampling_weights"].keys())

    # Initialize columns for each dataset
    sampling_weights = {name: [] for name in all_dataset_names}

    # Build columns
    for entry in raw_metrics:
        steps.append(entry.get("step", 0))
        timestamps.append(entry.get("ts", ""))

        # Fill in sampling weights, using None for missing values
        entry_weights = entry.get("sampling_weights", {})
        for name in all_dataset_names:
            sampling_weights[name].append(entry_weights.get(name, None))

    return {
        "steps": steps,
        "timestamps": timestamps,
        "sampling_weights": sampling_weights,
    }
