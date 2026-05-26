"""Per-run spec snapshot persistence.

The Identity tab inspects the live model via /api/spec. To inspect *other*
runs we snapshot the same payload to build/runs/<hash>/spec.json at startup
so it can be loaded later without re-instantiating the model.
"""

import io
import json
import os
import subprocess
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Optional

SPEC_SNAPSHOT_FILENAME = "spec.json"


def _serialise_args(args, excluded_attrs: set) -> dict:
    """Convert argparse.Namespace to a JSON-safe dict."""
    out = {}
    for key, value in vars(args).items():
        if key in excluded_attrs:
            continue
        try:
            json.dumps(value)
            out[key] = value
        except (TypeError, ValueError):
            out[key] = str(value)
    return out


def _capture_model_architecture(generator) -> Optional[str]:
    if not generator or not hasattr(generator, "model"):
        return None
    try:
        f = io.StringIO()
        with redirect_stdout(f):
            print(generator.model)
        return f.getvalue()
    except Exception as e:
        return f"Error getting model architecture: {e}"


def _capture_commit_timestamp() -> Optional[int]:
    try:
        result = subprocess.run(
            ["git", "show", "-s", "--format=%ct", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


def build_spec_payload(
    *,
    generator,
    truncated_hash: str,
    full_hash: str,
    param_stats: Optional[dict] = None,
    command: Optional[str] = None,
    timestamp: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict:
    """Build the spec payload that /api/spec returns for the live model.

    Request-dependent fields (git_url, masked_git_url) are intentionally
    omitted - the route layer fills those in for the live request, and
    they are meaningless for snapshots of other runs.
    """
    try:
        from praxis.cli import get_cli_args, get_loader_flag_attrs

        args = get_cli_args()
        excluded_attrs = get_loader_flag_attrs()
    except Exception:
        import argparse

        args = argparse.Namespace()
        excluded_attrs = set()

    return {
        "truncated_hash": truncated_hash or "unknown",
        "full_hash": full_hash or "unknown",
        "args": _serialise_args(args, excluded_attrs),
        "model_architecture": _capture_model_architecture(generator),
        "param_stats": param_stats or {},
        "timestamp": timestamp,
        "command": command,
        "seed": seed,
        "commit_timestamp": _capture_commit_timestamp(),
    }


def save_run_spec(run_dir: str, payload: dict) -> None:
    """Write the spec payload to <run_dir>/spec.json."""
    path = Path(run_dir) / SPEC_SNAPSHOT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def snapshot_run_spec(cfg, run, generator, param_stats, services) -> None:
    """Snapshot this run's spec to <run_dir>/spec.json (rank 0, best-effort).

    Lets the Identity tab inspect the run later, after the process exits.
    """
    if cfg.local_rank != 0:
        return
    try:
        api_server = services.api_server
        payload = build_spec_payload(
            generator=generator,
            truncated_hash=run.truncated_hash,
            full_hash=run.full_hash,
            param_stats=param_stats,
            command=run.full_command,
            timestamp=(api_server.launch_timestamp if api_server is not None else None),
            seed=cfg.seed,
        )
        save_run_spec(run.cache_dir, payload)
    except Exception:
        pass


def load_run_spec(run_dir: str) -> Optional[dict]:
    """Read <run_dir>/spec.json if present."""
    path = Path(run_dir) / SPEC_SNAPSHOT_FILENAME
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None
