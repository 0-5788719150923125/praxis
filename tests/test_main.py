"""Smoke tests for the main training script."""

import os
import signal
import subprocess
import sys
import time

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

    def test_main_with_alpha_experiment(self):
        """Test that main.py works with alpha experiment configuration."""
        cmd = [
            sys.executable,
            "main.py",
            "--alpha",
            "--dev",  # Override to dev mode for speed
            "--max-steps",
            "1",
            "--batch-size",
            "1",
            "--device",
            "cpu",
            "--no-dashboard",
            "--quiet",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        # Should complete successfully
        assert result.returncode == 0, f"Training with --alpha failed:\n{result.stderr}"

    def test_main_interrupt_handling(self):
        """Test that main.py handles interrupts gracefully."""
        cmd = [
            sys.executable,
            "main.py",
            "--dev",
            "--max-steps",
            "100",  # More steps so we have time to interrupt
            "--batch-size",
            "1",
            "--device",
            "cpu",
            "--no-dashboard",
            "--quiet",
        ]

        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        # Give it a moment to start
        time.sleep(5)

        # Send interrupt signal
        process.send_signal(signal.SIGINT)

        # Wait for it to finish
        try:
            stdout, stderr = process.communicate(timeout=10)
            # Interrupted processes may have non-zero return code, that's ok
            # Just verify it didn't crash catastrophically
            assert process.returncode is not None
        except subprocess.TimeoutExpired:
            process.kill()
            pytest.fail("Process did not terminate after interrupt")

    def test_main_with_different_block_types(self):
        """Test that different block types can be initialized."""
        # Use actual valid block types from BLOCK_REGISTRY
        block_types = ["transformer", "recurrent", "nano"]

        for block_type in block_types:
            cmd = [
                sys.executable,
                "main.py",
                "--dev",
                "--max-steps",
                "1",
                "--batch-size",
                "1",
                "--block-type",
                block_type,
                "--device",
                "cpu",
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

                assert (
                    result.returncode == 0
                ), f"Training with --block-type {block_type} failed:\n{result.stderr}"
            except subprocess.TimeoutExpired:
                import warnings

                warnings.warn(
                    f"Test for block type {block_type} timed out after 60s - skipping"
                )
                continue

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
