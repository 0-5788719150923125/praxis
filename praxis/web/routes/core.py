"""Core API routes."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import yaml
from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    make_response,
    render_template,
    request,
)
from praxis.optimizers import get_parameter_stats
from praxis.utils import mask_git_url

from ..config import CSP_POLICY
from ..spec_data import build_spec_payload, load_run_spec

core_bp = Blueprint("core", __name__)


def _compute_git_url(app_config) -> str | None:
    """Build the git URL for the *current* run based on the request host."""
    ngrok_url = app_config.get("ngrok_url")
    ngrok_secret = app_config.get("ngrok_secret")
    configured_host = app_config.get("configured_host")
    configured_port = app_config.get("configured_port")

    if ngrok_url and ngrok_secret:
        return f"{ngrok_url}/{ngrok_secret}/praxis"
    if configured_host and configured_host != "localhost":
        if configured_host.startswith(("https://", "http://")):
            return f"{configured_host}/praxis"
        return f"http://{configured_host}:{configured_port}/praxis"

    host = request.host.split(":")[0] if ":" in request.host else request.host
    if (
        host.endswith(".ngrok-free.app")
        or host.endswith(".ngrok.io")
        or host.endswith(".src.eco")
    ):
        return f"https://{host}/praxis"
    port = request.host.split(":")[1] if ":" in request.host else "80"
    return f"http://{host}:{port}/praxis"


@core_bp.route("/", methods=["GET"])
def home():
    """Serve the main page."""
    response = make_response(render_template("index.html"))
    response.headers["Content-Security-Policy"] = CSP_POLICY
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response


@core_bp.route("/api/ping", methods=["GET", "POST"])
def ping():
    """Simple endpoint to test if API is accessible."""
    response = jsonify({"status": "ok", "message": "Praxis API server is running"})
    return response


@core_bp.route("/api/spec", methods=["GET"])
def get_spec():
    """Get model specification including hashes and CLI arguments."""
    try:
        current_hash = current_app.config.get("truncated_hash")
        requested_hash = request.args.get("runs") or None

        # Snapshot path: serve a saved spec from build/runs/<hash>/spec.json
        # so the Identity tab can inspect runs other than the live one.
        if requested_hash and requested_hash != current_hash:
            run_dir = Path("build/runs") / requested_hash
            spec = load_run_spec(str(run_dir))
            if spec is None:
                spec = {
                    "truncated_hash": requested_hash,
                    "full_hash": None,
                    "args": {},
                    "model_architecture": None,
                    "param_stats": {},
                    "timestamp": None,
                    "command": None,
                    "seed": None,
                    "commit_timestamp": None,
                    "snapshot_missing": True,
                }
            spec["is_snapshot"] = True
            spec["is_current"] = False
            response = jsonify(spec)
            return response

        # Live path: build a fresh payload from app state and attach
        # request-dependent fields (git_url) on top.
        if not current_hash:
            current_hash = "unknown"
        full_hash = current_app.config.get("full_hash")
        if current_hash != "unknown" and not full_hash:
            full_hash = hashlib.sha256(current_hash.encode()).hexdigest()

        param_stats = current_app.config.get("param_stats", {})
        generator = current_app.config.get("generator")
        if not param_stats:
            try:
                if generator and hasattr(generator, "model"):
                    param_stats = get_parameter_stats(generator.model)
            except Exception:
                param_stats = {}

        spec = build_spec_payload(
            generator=generator,
            truncated_hash=current_hash,
            full_hash=full_hash or "unknown",
            param_stats=param_stats,
            command=current_app.config.get("launch_command"),
            timestamp=current_app.config.get("launch_timestamp"),
            seed=current_app.config.get("seed"),
        )

        git_url = _compute_git_url(current_app.config)
        spec["git_url"] = git_url
        spec["masked_git_url"] = mask_git_url(git_url) if git_url else None
        spec["is_snapshot"] = False
        spec["is_current"] = True

        response = jsonify(spec)
        return response

    except Exception as e:
        error_response = jsonify({"error": str(e)})
        return error_response, 500


@core_bp.route("/api/config", methods=["GET"])
def get_config():
    """Get current experiment configuration as YAML.

    Returns the active, running experiment config file from disk.
    No parameters accepted - returns only the current published config.
    """
    try:
        # Get the config file path from app config
        config_file = current_app.config.get("config_file")

        if not config_file:
            return Response("No experiment config file found", status=404)

        # Read the actual YAML file from disk
        config_path = Path(config_file)
        if not config_path.exists():
            return Response(f"Config file not found: {config_file}", status=404)

        # Resolve `extends` chain so published config is fully rendered
        config_data = load_rendered_config(config_path)

        def sort_dict_recursively(obj):
            """Recursively sort dictionary keys alphabetically."""
            if isinstance(obj, dict):
                return {k: sort_dict_recursively(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, list):
                return [sort_dict_recursively(item) for item in obj]
            else:
                return obj

        sorted_config = sort_dict_recursively(config_data)

        # Dump back to YAML with sorted keys
        yaml_content = yaml.dump(
            sorted_config, default_flow_style=False, sort_keys=False
        )

        response = Response(yaml_content, mimetype="text/yaml")
        response.headers.add(
            "Content-Disposition", f"attachment; filename={config_path.name}"
        )
        return response

    except Exception as e:
        error_response = Response(f"Error reading config: {str(e)}", status=500)
        return error_response
