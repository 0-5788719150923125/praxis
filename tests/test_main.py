"""End-to-end launches of main.py, the training entrypoint.

Each run starts from a scratch working directory, so nothing it writes (build/,
LICENSE, caches) lands in the repo; ``environments/`` is copied in so ``--dev``
still applies ``environments/dev.yml``. Both tests spawn a fresh interpreter
and are marked ``slow`` (run them with ``pytest -m slow``).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[1]


def _run_main(cwd, *args, timeout):
    shutil.copytree(REPO / "environments", cwd / "environments")
    return subprocess.run(
        [sys.executable, str(REPO / "main.py"), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )


@pytest.mark.network
def test_main_smoke_test_dev_mode(tmp_path):
    """One CPU training step in dev mode (Hugging Face data) exits cleanly."""
    result = _run_main(
        tmp_path,
        "--dev",
        "--max-steps",
        "1",
        "--batch-size",
        "1",
        "--device",
        "cpu",
        "--no-dashboard",
        "--quiet",
        "--no-docs",
        "--no-paper",
        timeout=180,
    )
    assert result.returncode == 0, f"Training failed:\n{result.stderr[-4000:]}"


def test_main_help_argument(tmp_path):
    result = _run_main(tmp_path, "--help", timeout=60)
    assert result.returncode == 0, result.stderr[-4000:]
    assert "--max-steps" in result.stdout
    assert "--dev" in result.stdout
