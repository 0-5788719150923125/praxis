"""Metrics API routes with intelligent downsampling and caching."""

import json
import hashlib
from pathlib import Path
from flask import Blueprint, jsonify, request, current_app
from typing import List, Dict, Any, Optional
from datetime import datetime

metrics_bp = Blueprint("metrics", __name__)


@metrics_bp.route("/api/metrics", methods=["GET", "OPTIONS"])
def get_metrics():
    """Get training metrics with smart downsampling and caching.

    Query Parameters:
        since: Only return metrics after this step (default: 0)
        limit: Maximum number of data points to return (default: 1000)
        downsample: Downsampling method - 'uniform', 'last', 'none' (default: 'uniform')

    Response Headers:
        ETag: Hash of metrics file for caching
        Cache-Control: max-age=5 (cache for 5 seconds)

    Returns:
        200: Metrics data
        304: Not Modified (if ETag matches)
        404: No metrics file found
        500: Server error
    """
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, If-None-Match")
        response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        return response

    try:
        # Get query parameters
        since_step = int(request.args.get('since', 0))
        limit = int(request.args.get('limit', 1000))
        downsample_method = request.args.get('downsample', 'uniform')
        runs_param = request.args.get('runs', '')  # Comma-separated hashes

        # Get current run directory
        current_hash = current_app.config.get("truncated_hash", "unknown")

        # Parse runs to fetch
        if runs_param:
            run_hashes = [h.strip() for h in runs_param.split(',') if h.strip()]
        else:
            run_hashes = [current_hash]

        # Fetch metrics for each run
        all_runs_data = []
        etag_parts = []

        for run_hash in run_hashes:
            run_dir = Path("build/runs") / run_hash
            metrics_file = run_dir / "metrics.jsonl"

            if not metrics_file.exists():
                continue

            # Calculate ETag component
            stat = metrics_file.stat()
            etag_parts.append(f"{run_hash}:{stat.st_mtime}:{stat.st_size}")

            # Read and parse metrics
            raw_metrics = _read_metrics_file(metrics_file, since_step)

            if not raw_metrics:
                continue

            # Downsample if needed
            if len(raw_metrics) > limit:
                raw_metrics = _downsample_metrics(raw_metrics, limit, downsample_method)

            # Transform to API format
            metrics_data = _transform_metrics(raw_metrics)

            all_runs_data.append({
                "hash": run_hash,
                "is_current": run_hash == current_hash,
                "metrics": metrics_data,
                "metadata": {
                    "model_hash": run_hash,
                    "last_updated": stat.st_mtime,
                    "num_points": len(raw_metrics),
                    "downsampled": len(raw_metrics) < limit,
                    "first_step": raw_metrics[0]["step"] if raw_metrics else 0,
                    "last_step": raw_metrics[-1]["step"] if raw_metrics else 0
                }
            })

        if not all_runs_data:
            return jsonify({
                "status": "no_data",
                "message": "No metrics found for the requested runs"
            }), 404

        # Calculate combined ETag
        etag = hashlib.md5('|'.join(etag_parts).encode()).hexdigest()

        # Check if client has cached version
        if request.headers.get('If-None-Match') == etag:
            response = jsonify({"status": "not_modified"})
            response.headers['ETag'] = etag
            return response, 304

        # Build response
        response_data = {
            "status": "ok",
            "source": "metrics_logger",
            "runs": all_runs_data,
            "metadata": {
                "total_params": current_app.config.get("total_params", "N/A"),
                "current_hash": current_hash,
                "num_runs": len(all_runs_data)
            }
        }

        response = jsonify(response_data)
        response.headers['ETag'] = etag
        response.headers['Cache-Control'] = 'max-age=5'  # Cache for 5 seconds
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        error_response = jsonify({"error": str(e), "status": "error"})
        error_response.headers.add("Access-Control-Allow-Origin", "*")
        return error_response, 500


def _read_metrics_file(filepath: Path, since_step: int = 0) -> List[Dict[str, Any]]:
    """Read metrics from JSONL file, filtering by step.

    Args:
        filepath: Path to metrics.jsonl file
        since_step: Only include metrics after this step

    Returns:
        List of metric dictionaries
    """
    metrics = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                if entry.get("step", 0) >= since_step:
                    metrics.append(entry)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue

    return metrics


def _downsample_metrics(
    metrics: List[Dict[str, Any]],
    target_size: int,
    method: str = 'uniform'
) -> List[Dict[str, Any]]:
    """Downsample metrics to target size.

    Args:
        metrics: List of metric dictionaries
        target_size: Target number of points
        method: Downsampling method ('uniform', 'last', 'none')

    Returns:
        Downsampled list of metrics
    """
    if len(metrics) <= target_size or method == 'none':
        return metrics

    if method == 'last':
        # Keep only the last N points
        return metrics[-target_size:]

    elif method == 'uniform':
        # Uniform sampling: always keep first and last, sample uniformly in between
        if target_size < 2:
            return [metrics[-1]]

        downsampled = [metrics[0]]  # Always keep first

        # Calculate indices for uniform sampling
        step_size = (len(metrics) - 1) / (target_size - 1)
        indices = [int(i * step_size) for i in range(1, target_size - 1)]

        for idx in indices:
            downsampled.append(metrics[idx])

        downsampled.append(metrics[-1])  # Always keep last

        return downsampled

    else:
        # Default to uniform
        return _downsample_metrics(metrics, target_size, 'uniform')


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
    if 'step' in result:
        result['steps'] = result.pop('step')

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
            return jsonify({
                "status": "ok",
                "runs": []
            })

        runs = []
        current_hash = current_app.config.get("truncated_hash", "unknown")

        # Scan all run directories
        for run_path in runs_dir.iterdir():
            if not run_path.is_dir():
                continue

            truncated_hash = run_path.name
            config_file = run_path / "config.json"
            metrics_file = run_path / "metrics.jsonl"

            # Skip if no metrics file
            if not metrics_file.exists():
                continue

            # Load config if available
            config = {}
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                except:
                    pass

            # Get metrics file stats
            stat = metrics_file.stat()

            # Count lines in metrics file to get step count
            num_steps = 0
            try:
                with open(metrics_file, 'r') as f:
                    num_steps = sum(1 for line in f if line.strip())
            except:
                pass

            runs.append({
                "hash": truncated_hash,
                "is_current": truncated_hash == current_hash,
                "command": config.get("command", "unknown"),
                "created": config.get("created"),
                "last_updated": config.get("last_updated"),
                "metrics_updated": stat.st_mtime,
                "num_steps": num_steps,
                "has_metrics": metrics_file.exists()
            })

        # Sort by last updated (most recent first)
        runs.sort(key=lambda x: x.get("metrics_updated", 0), reverse=True)

        response_data = {
            "status": "ok",
            "runs": runs,
            "current_hash": current_hash
        }

        response = jsonify(response_data)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        error_response = jsonify({"error": str(e), "status": "error"})
        error_response.headers.add("Access-Control-Allow-Origin", "*")
        return error_response, 500
