#!/usr/bin/env python3
"""
Script to run the praxis training with hard-coded alpha configuration.

This script runs run.py with predefined arguments for the alpha setup.
"""

import os
import sys
from pathlib import Path


def main():
    """Run the training script with alpha configuration."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    run_script = script_dir / "run.py"

    # Hard-coded arguments for alpha configuration
    args = [
        sys.executable,  # Use the same Python interpreter
        str(run_script),
        "--seed",
        "42",
        "--device",
        "cuda:0",
        "--batch-size",
        "16",
        "--depth",
        "3",
        "--num-experts",
        "3",
        "--vocab-size",
        "4096",
        "--tie-weights",
        "--schedule-free",
        "--wandb",
        "--rl-type",
        "cot",
    ]

    print(f"Running: {' '.join(args)}")

    # Use os.execv to replace this process entirely
    # This ensures proper signal handling and terminal control
    os.execv(sys.executable, args)


if __name__ == "__main__":
    main()
