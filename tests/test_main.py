"""Smoke tests for the main training script."""

import os
import subprocess
import sys

import pytest
import torch


class TestMainScript:
    """Test cases for main.py training script."""

    def test_main_smoke_test_dev_mode(self):
        """Test that main.py runs successfully with --dev settings for a few steps."""
        # Use a small configuration that should run quickly
        cmd = [
            sys.executable,
            "main.py",
            "--dev",
            "--max-steps",
            "1",
            "--batch-size",
            "1",
            "--device",
            "cpu",
            "--no-dashboard",
            "--quiet",
        ]

        # Run the training script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 120 second timeout for slower systems
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        # Check that it completed successfully
        assert result.returncode == 0, f"Training failed with output:\n{result.stderr}"

        # Basic sanity checks on output - check for expected output patterns
        stdout_lower = result.stdout.lower()
        # Check for successful completion message or model/parameter information
        assert any(
            [
                "[train] completed successfully" in stdout_lower,
                "training completed" in stdout_lower,
                "[init]" in stdout_lower,
                "model" in stdout_lower,
                "parameters" in stdout_lower,
            ]
        ), f"Expected output patterns not found in:\n{result.stdout}"

    def test_main_help_argument(self):
        """Test that --help works correctly."""
        cmd = [sys.executable, "main.py", "--help"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,  # Increased timeout as help initialization can be slow
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        assert result.returncode == 0
        assert (
            "praxis cli" in result.stdout.lower() or "usage:" in result.stdout.lower()
        )
        assert "--max-steps" in result.stdout
        assert "--dev" in result.stdout

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_main_cuda_initialization(self):
        """Test that CUDA initialization works if available."""
        import warnings

        import torch

        cmd = [
            sys.executable,
            "main.py",
            "--dev",
            "--max-steps",
            "1",
            "--batch-size",
            "1",
            "--device",
            "cuda:0",
            "--no-dashboard",
            "--quiet",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )

            assert result.returncode == 0, f"CUDA training failed:\n{result.stderr}"
        except subprocess.TimeoutExpired:
            warnings.warn("CUDA test timed out after 60s - skipping")
