#!/usr/bin/env python3
"""
Script to run the praxis training with hard-coded alpha configuration.

This script provides a base "alpha" configuration but allows any argument
to be overridden via command line. Uses the same CLI interface as run.py.
"""

import os
import sys
from pathlib import Path


def main():
    """Run the training script with alpha configuration, allowing CLI overrides."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    run_script = script_dir / "run.py"

    # Base alpha configuration (can be overridden by CLI args)
    alpha_defaults = [
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
        "--attention-type",
        "standard",
        "--strategy",
        "naive",
        "--tie-weights",
        "--schedule-free",
    ]

    # Get user-provided arguments (everything passed to this script)
    user_args = sys.argv[1:]

    # Merge alpha defaults with user overrides
    final_args = merge_args(alpha_defaults, user_args)

    # Build the complete command
    args = [sys.executable, str(run_script)] + final_args

    # Use os.execv to replace this process entirely
    # This ensures proper signal handling and terminal control
    os.execv(sys.executable, args)


def merge_args(defaults, overrides):
    """
    Merge default arguments with user overrides.

    User arguments take precedence over defaults.
    Handles both flag arguments (--flag) and value arguments (--key value).

    Args:
        defaults: List of default arguments (e.g., ["--batch-size", "16", "--wandb"])
        overrides: List of user-provided arguments to override defaults

    Returns:
        List of merged arguments
    """
    # Parse defaults into a dictionary
    default_dict = {}
    i = 0
    while i < len(defaults):
        arg = defaults[i]
        if arg.startswith("-"):
            # Check if next item is a value or another flag
            if i + 1 < len(defaults) and not defaults[i + 1].startswith("-"):
                # This is a key-value pair
                default_dict[arg] = defaults[i + 1]
                i += 2
            else:
                # This is a flag
                default_dict[arg] = None
                i += 1
        else:
            i += 1

    # Parse overrides and update defaults
    override_dict = default_dict.copy()
    i = 0
    while i < len(overrides):
        arg = overrides[i]
        if arg.startswith("-"):
            # Check if next item is a value or another flag
            if i + 1 < len(overrides) and not overrides[i + 1].startswith("-"):
                # This is a key-value pair
                override_dict[arg] = overrides[i + 1]
                i += 2
            else:
                # This is a flag
                override_dict[arg] = None
                i += 1
        else:
            i += 1

    # Convert back to list format
    result = []
    for key, value in override_dict.items():
        result.append(key)
        if value is not None:
            result.append(value)

    return result


if __name__ == "__main__":
    main()
