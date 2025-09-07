"""System utilities for process management and updates."""

import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from glob import glob


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
    ckpt_dir = os.path.join(cache_dir, "model")

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


def perform_reset(cache_dir, truncated_hash, integration_loader=None):
    """Perform a full reset of the project, clearing all cached data.

    Args:
        cache_dir: The cache directory to clean
        truncated_hash: The hash identifying this project instance
        integration_loader: Optional integration loader for getting additional cleanup directories
    """

    grace_time = 7

    print()
    print(f"    WARNING: Resetting project {truncated_hash}")
    print(f" ⚠️ This will permanently delete all checkpoints and cached data.")
    print(f"    Press Ctrl+C within {grace_time} seconds to cancel...")

    try:
        time.sleep(grace_time)
    except KeyboardInterrupt:
        print("\n ✓   Reset cancelled.")
        sys.exit(0)

    print("\n 🗑️ Performing reset...")

    # Get directories to clean
    directories = ["logs"]
    if integration_loader:
        directories.extend(integration_loader.get_cleanup_directories())

    # Clean directories
    for directory in directories:
        dir_path = os.path.join(cache_dir, directory)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)
            print(f"    Removed: {directory}/")

    # Clean checkpoint files
    ckpt_pattern = os.path.join(cache_dir, "model", "*.ckpt")
    checkpoints = glob(ckpt_pattern)
    for checkpoint in checkpoints:
        try:
            os.remove(checkpoint)
            print(f"    Removed: {os.path.basename(checkpoint)}")
        except Exception:
            pass

    time.sleep(1)
    print("\n ✓  Reset complete.\n")


def show_launch_animation(model, truncated_hash):
    """Display the fancy launch animation for model loading.

    Args:
        model: The model to display
        truncated_hash: The hash identifying this instance
    """
    plan = str(model.__repr__).splitlines()
    launch_duration = random.uniform(6.7, 7.3)
    acceleration_curve = random.uniform(3.5, 4.5)
    start_time = time.time()

    time.sleep(max(0, random.gauss(1.0, 3.0)))

    for i, line in enumerate(plan):
        print(line)
        progress = i / len(plan)
        scale_factor = launch_duration * (acceleration_curve + 1) / len(plan)
        delay = scale_factor * (progress**acceleration_curve)
        time.sleep(delay)

    elapsed_time = time.time() - start_time
    print(f"Loaded: {truncated_hash} in {elapsed_time:.3f} seconds.")
    time.sleep(2)
