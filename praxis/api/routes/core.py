"""Core API routes."""

import hashlib
import io
import json
import os
import subprocess
from contextlib import redirect_stdout
from datetime import datetime

import yaml
from flask import Blueprint, Response, jsonify, make_response, render_template, request

from ..config import CSP_POLICY

core_bp = Blueprint("core", __name__)


@core_bp.route("/", methods=["GET"])
def home():
    """Serve the main page."""
    response = make_response(render_template("index.html"))
    response.headers["Content-Security-Policy"] = CSP_POLICY
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response


@core_bp.route("/api/ping", methods=["GET", "POST", "OPTIONS"])
def ping():
    """Simple endpoint to test if API is accessible."""
    response = jsonify({"status": "ok", "message": "Praxis API server is running"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return response


@core_bp.route("/api/spec", methods=["GET", "OPTIONS"])
def get_spec():
    """Get model specification including hashes and CLI arguments."""
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        return response

    try:
        from flask import current_app

        # Try to get CLI args, but handle the case where they're not available
        try:
            from praxis.cli import get_cli_args

            args = get_cli_args()
        except:
            # If CLI args aren't available (e.g., in tests), use empty namespace
            import argparse

            args = argparse.Namespace()

        # Convert args to dict, filtering out non-serializable items
        args_dict = {}
        for key, value in vars(args).items():
            try:
                json.dumps(value)
                args_dict[key] = value
            except (TypeError, ValueError):
                args_dict[key] = str(value)

        # Use the hashes from app config
        truncated_hash = current_app.config.get("truncated_hash")
        full_hash = current_app.config.get("full_hash")

        # Fallback for backward compatibility
        if truncated_hash and not full_hash:
            full_hash = hashlib.sha256(truncated_hash.encode()).hexdigest()

        if not truncated_hash:
            truncated_hash = "unknown"
            full_hash = "unknown"

        # Get the model architecture string
        model_arch = None
        try:
            generator = current_app.config.get("generator")
            if generator and hasattr(generator, "model"):
                model = generator.model
                f = io.StringIO()
                with redirect_stdout(f):
                    print(model)
                model_arch = f.getvalue()
        except Exception as e:
            model_arch = f"Error getting model architecture: {str(e)}"

        # Use the simplified param_stats from the app config
        param_stats = current_app.config.get("param_stats", {})

        # If not available, try to calculate it
        if not param_stats:
            try:
                from praxis.optimizers import get_parameter_stats

                generator = current_app.config.get("generator")
                if generator and hasattr(generator, "model"):
                    model = generator.model
                    param_stats = get_parameter_stats(model)
            except:
                param_stats = {}

        # Get the launch command and timestamp
        command = current_app.config.get("launch_command")
        timestamp = current_app.config.get("launch_timestamp")

        # Get the appropriate git URL
        git_url = None
        ngrok_url = current_app.config.get("ngrok_url")
        ngrok_secret = current_app.config.get("ngrok_secret")

        if ngrok_url and ngrok_secret:
            git_url = f"{ngrok_url}/{ngrok_secret}/praxis"
        else:
            host = request.host.split(":")[0] if ":" in request.host else request.host
            if (
                host.endswith(".ngrok-free.app")
                or host.endswith(".ngrok.io")
                or host.endswith(".src.eco")
            ):
                git_url = f"https://{host}/praxis"
            else:
                port = request.host.split(":")[1] if ":" in request.host else "80"
                git_url = f"http://{host}:{port}/praxis"

        from praxis.utils import mask_git_url

        # Get commit timestamp
        commit_timestamp = None
        try:
            timestamp_result = subprocess.run(
                ["git", "show", "-s", "--format=%ct", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )
            if timestamp_result.returncode == 0:
                commit_timestamp = int(timestamp_result.stdout.strip())
        except:
            pass

        spec = {
            "truncated_hash": truncated_hash,
            "full_hash": full_hash,
            "args": args_dict,
            "model_architecture": model_arch,
            "param_stats": param_stats,
            "timestamp": timestamp,
            "command": command,
            "git_url": git_url,
            "masked_git_url": mask_git_url(git_url) if git_url else None,
            "seed": current_app.config.get("seed"),
            "commit_timestamp": commit_timestamp,
        }

        response = jsonify(spec)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        error_response = jsonify({"error": str(e)})
        error_response.headers.add("Access-Control-Allow-Origin", "*")
        return error_response, 500


@core_bp.route("/api/config", methods=["GET", "OPTIONS"])
def get_config():
    """Get current experiment configuration as YAML.

    Returns the active, running experiment config file from disk.
    No parameters accepted - returns only the current published config.
    """
    if request.method == "OPTIONS":
        response = Response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        return response

    try:
        from pathlib import Path

        from flask import current_app

        # Get the config file path from app config
        config_file = current_app.config.get("config_file")

        if not config_file:
            return Response("No experiment config file found", status=404)

        # Read the actual YAML file from disk
        config_path = Path(config_file)
        if not config_path.exists():
            return Response(f"Config file not found: {config_file}", status=404)

        # Load YAML and sort keys recursively
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

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
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add(
            "Content-Disposition", f"attachment; filename={config_path.name}"
        )
        return response

    except Exception as e:
        error_response = Response(f"Error reading config: {str(e)}", status=500)
        error_response.headers.add("Access-Control-Allow-Origin", "*")
        return error_response
