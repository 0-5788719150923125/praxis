"""System utilities for process management and updates."""

import os
import re
import signal
import subprocess
import sys


def sigint_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) by killing all spawned processes."""
    print("\nCtrl+C detected. Killing all spawned processes.")
    # Kill the entire process group
    os.killpg(os.getpgid(0), signal.SIGTERM)
    sys.exit(1)


def check_for_updates():
    """Check if the git repository has updates available."""
    try:
        # First, fetch the latest changes from remote
        subprocess.run(["git", "fetch"], check=True, capture_output=True)

        # Try to get the current branch name
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        # Get commit counts ahead and behind
        result = subprocess.run(
            ["git", "rev-list", "--count", "--left-right", f"origin/{branch}...HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )

        behind, ahead = map(int, result.stdout.strip().split())

        # Only print if we're behind
        if behind > 0:
            # Get the latest remote commit info
            latest_commit_info = subprocess.run(
                ["git", "log", f"origin/{branch}", "-1", "--pretty=format:%h - %s"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            print("=" * 50)
            print(f"🔄 UPDATE AVAILABLE: {behind} commit(s) behind origin/{branch}")
            print(f"   Latest: {latest_commit_info}")
            print(f"   Run 'git pull' to update")
            print("=" * 50)

    except subprocess.CalledProcessError:
        # Silently fail if we're not in a git repo or other git issues
        pass
    except Exception:
        # Silently fail for any other errors
        pass


def find_latest_checkpoint(cache_dir):
    """Find the latest checkpoint file in the cache directory."""
    # Construct the checkpoint directory path
    ckpt_dir = os.path.join(cache_dir, "praxis")

    # Get all checkpoint files
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]

    if not ckpt_files:
        return None

    # Extract batch numbers using regex
    # This will find numbers after "batch=" and before ".ckpt"
    batch_numbers = []
    for filename in ckpt_files:
        match = re.search(r"batch=(\d+)\.0\.ckpt", filename)
        if match:
            batch_numbers.append((int(match.group(1)), filename))

    if not batch_numbers:
        return None

    # Find the file with the largest batch number
    latest_batch = max(batch_numbers, key=lambda x: x[0])
    latest_checkpoint = os.path.join(ckpt_dir, latest_batch[1])

    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def initialize_lazy_modules(model, device):
    """Initialize lazy modules in a model by doing a dummy forward pass."""
    import torch
    
    model = model.to(device)

    # Create dummy batch for initialization
    batch_size = 2
    seq_length = 64
    dummy_input = torch.ones((batch_size, seq_length), dtype=torch.long).to(device)
    dummy_labels = dummy_input[..., 1:].contiguous()

    # Do a dummy forward pass to initialize lazy parameters
    model.train()
    outputs = model(input_ids=dummy_input, labels=dummy_labels)

    # Reset any gradient accumulation
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

    return model