"""Run management system for namespace-based experiment tracking."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class RunManager:
    """Manages experiment runs with automatic namespacing by hash."""

    def __init__(self, base_dir: str = "./build"):
        """Initialize the run manager.

        Args:
            base_dir: Base directory for all builds (default: ./build)
        """
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.shared_dir = self.base_dir / "shared"
        self.current_link = self.base_dir / "current"

        # Ensure directories exist
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.shared_dir.mkdir(parents=True, exist_ok=True)

    def get_run_dir(self, truncated_hash: str) -> Path:
        """Get the directory path for a specific run.

        Args:
            truncated_hash: The truncated hash identifying the run

        Returns:
            Path to the run directory
        """
        return self.runs_dir / truncated_hash

    def setup_run(self, truncated_hash: str, full_command: str, args_hash: str,
                  preserve: bool = False) -> Tuple[Path, bool]:
        """Set up a run directory and return its path.

        Args:
            truncated_hash: The truncated hash for this run
            full_command: The full command line used
            args_hash: The full argument hash
            preserve: Whether to mark this run as preserved

        Returns:
            Tuple of (run_directory_path, is_existing_run)
        """
        run_dir = self.get_run_dir(truncated_hash)
        is_existing = run_dir.exists()

        # Create run directory structure
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "model").mkdir(exist_ok=True)
        (run_dir / "wandb").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)

        # Load existing config if present
        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}

        # Update config (preserve existing preserve flag if set)
        config.update({
            "truncated_hash": truncated_hash,
            "full_hash": args_hash,
            "command": full_command,
            "last_updated": datetime.now().isoformat(),
            "preserve": config.get("preserve", False) or preserve
        })

        # If this is a new run, add creation time
        if not is_existing:
            config["created"] = datetime.now().isoformat()

        # Save config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Update symlink to current run
        self._update_current_link(truncated_hash)

        return run_dir, is_existing

    def _update_current_link(self, truncated_hash: str):
        """Update the 'current' symlink to point to the active run.

        Args:
            truncated_hash: The hash of the currently active run
        """
        run_dir = self.get_run_dir(truncated_hash)

        # Remove existing symlink if it exists
        if self.current_link.exists() or self.current_link.is_symlink():
            self.current_link.unlink()

        # Create new symlink (relative path for portability)
        relative_path = Path("runs") / truncated_hash
        self.current_link.symlink_to(relative_path)

    def list_runs(self) -> List[Dict]:
        """List all available runs with their metadata.

        Returns:
            List of run information dictionaries
        """
        runs = []

        if not self.runs_dir.exists():
            return runs

        for run_path in sorted(self.runs_dir.iterdir()):
            if not run_path.is_dir():
                continue

            config_path = run_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                # Basic info for runs without config
                config = {
                    "truncated_hash": run_path.name,
                    "preserve": False
                }

            # Add directory size
            size = sum(f.stat().st_size for f in run_path.rglob('*') if f.is_file())
            config['size_bytes'] = size
            config['size_human'] = self._format_bytes(size)

            # Check if this is the current run
            if self.current_link.exists():
                try:
                    current_target = self.current_link.resolve().name
                    config['is_current'] = (current_target == run_path.name)
                except:
                    config['is_current'] = False
            else:
                config['is_current'] = False

            runs.append(config)

        return runs

    def reset_run(self, truncated_hash: str, force: bool = False) -> bool:
        """Reset (delete) a specific run by hash.

        The preserve flag only protects runs from bulk/general resets.
        When explicitly targeting a hash (by running with same config + --reset),
        we always allow the reset since the user is being explicit.

        Args:
            truncated_hash: The hash of the run to reset
            force: If True, always delete even if preserved (used for explicit resets)

        Returns:
            True if the run was deleted or didn't exist, False if preserved and not forced
        """
        run_dir = self.get_run_dir(truncated_hash)

        if not run_dir.exists():
            # Run doesn't exist, nothing to reset
            return True

        # Check if run is preserved (only matters for bulk operations, not explicit)
        if not force:
            config_path = run_dir / "config.json"
            is_preserved = False
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    is_preserved = config.get("preserve", False)

            if is_preserved:
                print(f"Run {truncated_hash} is preserved and will not be reset")
                return False

        # Delete the run
        print(f"Resetting run: {truncated_hash}")
        shutil.rmtree(run_dir)

        # Update current link if needed
        if self.current_link.exists() or self.current_link.is_symlink():
            try:
                # Check if current link points to the deleted run
                if self.current_link.resolve().name == truncated_hash:
                    self.current_link.unlink()
            except:
                # Link is broken or problematic, remove it
                if self.current_link.is_symlink():
                    self.current_link.unlink()

        return True

    def get_cache_dir(self, truncated_hash: str) -> str:
        """Get the cache directory for a specific run.

        Args:
            truncated_hash: The truncated hash for the run

        Returns:
            String path to the cache directory
        """
        return str(self.get_run_dir(truncated_hash))

    def get_shared_dir(self, subdir: Optional[str] = None) -> str:
        """Get the shared directory or a subdirectory within it.

        Args:
            subdir: Optional subdirectory name within shared

        Returns:
            String path to the shared directory
        """
        if subdir:
            path = self.shared_dir / subdir
            path.mkdir(parents=True, exist_ok=True)
            return str(path)
        return str(self.shared_dir)

    @staticmethod
    def _format_bytes(bytes_num: int) -> str:
        """Format bytes into human readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if abs(bytes_num) < 1024.0:
                return f"{bytes_num:3.1f} {unit}"
            bytes_num /= 1024.0
        return f"{bytes_num:.1f} PB"

    def mark_preserve(self, truncated_hash: str, preserve: bool = True):
        """Mark a run as preserved or not.

        Args:
            truncated_hash: The hash of the run to modify
            preserve: Whether to preserve (True) or unpreserve (False)
        """
        run_dir = self.get_run_dir(truncated_hash)
        if not run_dir.exists():
            print(f"Run {truncated_hash} does not exist")
            return

        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}

        config["preserve"] = preserve
        config["preserve_updated"] = datetime.now().isoformat()

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        status = "preserved" if preserve else "unpreserved"
        print(f"Run {truncated_hash} marked as {status}")