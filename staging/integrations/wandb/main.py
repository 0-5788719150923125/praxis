"""Weights & Biases integration implementation for Praxis."""

import os
import re
import shutil
from datetime import datetime


def add_cli_args(parser):
    """Add wandb CLI arguments to the parser."""
    other_group = None

    # Find the 'other' argument group
    for group in parser._action_groups:
        if group.title == "other":
            other_group = group
            break

    if other_group is None:
        other_group = parser.add_argument_group("other")

    other_group.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Log metrics to Weights and Biases (https://wandb.ai)",
    )
    other_group.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Custom name for the W&B run (default: auto-generated)",
    )


def initialize(args, cache_dir, ckpt_path=None, truncated_hash=None):
    """Initialize wandb and return logger configuration."""
    import wandb

    wandb.login()

    # Configure wandb options
    wandb_opts = dict(
        project="praxis",
        save_dir=cache_dir,
        save_code=False,  # Don't save code files
        log_model=False,  # Don't log model artifacts
        settings=wandb.Settings(
            disable_code=True,  # Disable code saving
            disable_git=True,  # Disable git info
            _disable_stats=True,  # Disable system metrics
        ),
    )

    # Handle resumption from checkpoint
    if ckpt_path is not None:
        run_id = find_latest_wandb_run(cache_dir, cleanup_old_runs=True)
        if run_id is not None:
            wandb_opts["id"] = run_id
            wandb_opts["resume"] = "must"

    # Set run name
    if args.wandb_run_name:
        wandb_opts["name"] = args.wandb_run_name
    elif truncated_hash:
        wandb_opts["name"] = truncated_hash

    return {"wandb_opts": wandb_opts}


def cleanup():
    """Cleanup wandb resources."""
    try:
        import wandb

        wandb.finish()
    except Exception:
        pass


def create_logger(cache_dir, ckpt_path=None, truncated_hash=None, **kwargs):
    """Create and configure a wandb logger if wandb is enabled."""
    # Check if wandb is enabled through kwargs
    if not kwargs.get("wandb_enabled", False):
        return None

    # Get args from kwargs
    args = kwargs.get("args")
    if not args:
        return None

    # Initialize wandb
    init_result = initialize(args, cache_dir, ckpt_path, truncated_hash)
    if init_result and "wandb_opts" in init_result:
        return CustomWandbLogger(**init_result["wandb_opts"])

    return None


# Alias for backward compatibility
provide_logger = create_logger


def find_latest_wandb_run(cache_dir, cleanup_old_runs=True):
    """
    Find the latest wandb run directory, extract its ID, and clean up older run directories.
    """
    wandb_dir = os.path.join(cache_dir, "wandb")
    if not os.path.exists(wandb_dir):
        return None

    # Pattern for run directories: run-YYYYMMDD_HHMMSS-<random_id>
    run_pattern = re.compile(r"run-(\d{8}_\d{6})-([a-z0-9]+)")

    # Step 1: Find all run directories and identify the most recent one
    all_run_dirs = []
    latest_timestamp = None
    latest_run_id = None
    latest_dir_path = None

    for dirname in os.listdir(wandb_dir):
        dir_path = os.path.join(wandb_dir, dirname)
        if not os.path.isdir(dir_path):
            continue

        match = run_pattern.match(dirname)
        if match:
            timestamp_str = match.group(1)  # YYYYMMDD_HHMMSS
            run_id = match.group(2)  # random ID

            # Convert timestamp string to datetime object
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                # Store information about this run directory
                all_run_dirs.append((dir_path, timestamp, run_id))

                # Update if this is the most recent run
                if latest_timestamp is None or timestamp > latest_timestamp:
                    latest_timestamp = timestamp
                    latest_run_id = run_id
                    latest_dir_path = dir_path
            except ValueError:
                continue

    # Step 2: Clean up older run directories if requested and if we have more than one directory
    if cleanup_old_runs and len(all_run_dirs) > 1 and latest_dir_path is not None:
        print(
            f"Cleaning up old wandb run directories. Keeping only: {os.path.basename(latest_dir_path)}"
        )
        for dir_path, _, _ in all_run_dirs:
            if dir_path != latest_dir_path:
                try:
                    print(f"Removing old wandb run: {os.path.basename(dir_path)}")
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Error removing directory {dir_path}: {e}")

    # Step 3: Now try to find the run ID using both methods
    # First check the symbolic link method
    latest_run_path = os.path.join(wandb_dir, "latest-run")
    if os.path.exists(latest_run_path):
        pattern = re.compile(r"run-([a-z0-9]+)\.wandb")
        for filename in os.listdir(latest_run_path):
            match = pattern.match(filename)
            if match:
                print("resuming wandb from symbolic path")
                return match.group(1)

    # If symbolic link method didn't work, use the latest run ID we found earlier
    if latest_run_id:
        print("resuming wandb from full path")
        return latest_run_id

    # If we didn't find any valid run IDs
    return None


class CustomWandbLogger:
    """Custom WandB logger that handles hyperparameter logging properly."""

    def __init__(self, **wandb_opts):
        from lightning.pytorch.loggers import WandbLogger

        self._wandb_logger = WandbLogger(**wandb_opts)

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped logger."""
        return getattr(self._wandb_logger, name)

    def log_hyperparams(self, params):
        """Custom hyperparameter logging that flattens hparams."""
        # Create new dict with all non-hparams entries
        cleaned_params = {k: v for k, v in params.items() if k != "hparams"}

        # Update with contents of hparams dict if it exists
        if "hparams" in params:
            cleaned_params.update(params["hparams"])

        self._wandb_logger.log_hyperparams(cleaned_params)
