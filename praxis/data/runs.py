"""Run management system for namespace-based experiment tracking."""

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RunContext:
    """Identifies a prepared run directory and its hashes."""

    cache_dir: str  # the namespaced run directory
    run_dir: Path
    truncated_hash: str
    full_hash: str
    full_command: str
    is_existing_run: bool


def print_runs(base_cache_dir: str) -> int:
    """Print all known runs (the ``--list-runs`` shortcut). Returns exit code."""
    runs = RunManager(base_cache_dir).list_runs()
    if not runs:
        print("No runs found.")
        return 0

    print("\nAvailable runs:")
    for run in runs:
        status = "[CURRENT]" if run.get("is_current") else ""
        preserved = "[PRESERVED]" if run.get("preserve") else ""
        created = run.get("created", "Unknown")
        size = run.get("size_human", "Unknown")
        print(
            f"  {run['truncated_hash']} - {size} - Created: {created} {preserved} {status}"
        )
    return 0


def setup_training_run(cfg) -> RunContext:
    """Resolve this run's directory, handle --reset, and seed RNGs.

    Mirrors the original main.py preamble: log the command, reset the run
    when asked (explicit resets ignore the preserve flag), set up the
    namespaced directory, then seed everything.
    """
    from praxis.cli import log_command
    from praxis.trainers import seed_everything

    full_command, args_hash, truncated_hash = log_command()

    run_manager = RunManager(cfg.cache_dir)

    # Crash-loop breaker. With --reset-after N (N>0), count consecutive launches
    # that don't advance the checkpoint; once N is reached, force a reset so a
    # wedged checkpoint self-heals in a respawning (systemd) environment. A launch
    # that finds the checkpoint advanced since last time is treated as healthy and
    # resets the counter, so a stable run is never wiped by routine restarts.
    do_reset = cfg.reset
    reset_after = getattr(cfg, "reset_after", 0) or 0
    if not do_reset and reset_after > 0:
        signature = _checkpoint_signature(run_manager.get_run_dir(truncated_hash))
        attempts = run_manager.bump_reset_marker(truncated_hash, signature)
        if attempts >= reset_after:
            print(
                f"[RESET-AFTER] {attempts} consecutive launch(es) with no "
                f"checkpoint progress >= --reset-after={reset_after}; forcing --reset."
            )
            do_reset = True
            run_manager.clear_reset_marker(truncated_hash)
        else:
            print(
                f"[RESET-AFTER] launch attempt {attempts}/{reset_after} "
                f"(checkpoint unchanged increments this; reset on reaching the limit)."
            )

    if do_reset:
        run_manager.reset_run(truncated_hash, force=True)

    run_dir, is_existing_run = run_manager.setup_run(
        truncated_hash, full_command, args_hash, cfg.preserve
    )

    if is_existing_run:
        print(f"[RUN] Resuming existing run: {truncated_hash}")
    else:
        print(f"[RUN] Starting new run: {truncated_hash}")

    seed_everything(cfg.seed, workers=True)

    return RunContext(
        cache_dir=str(run_dir),
        run_dir=run_dir,
        truncated_hash=truncated_hash,
        full_hash=args_hash,
        full_command=full_command,
        is_existing_run=is_existing_run,
    )


def _checkpoint_signature(run_dir: Path) -> str:
    """Fingerprint the run's current resume checkpoint, so a crash loop (same
    checkpoint every launch) is distinguishable from healthy progress (it
    advanced). Returns "none" when there's nothing to resume from."""
    candidates = [
        run_dir / "model" / "last.ckpt",  # Lightning resume symlink
        run_dir / "mono_forward.pt",      # Mono-Forward trainer checkpoint
    ]
    for path in candidates:
        try:
            if path.exists():
                real = path.resolve()  # follow last.ckpt -> the real file
                return f"{real}:{int(real.stat().st_mtime)}"
        except OSError:
            continue
    return "none"


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

    def _reset_marker_path(self, truncated_hash: str) -> Path:
        """Path to a run's reset marker. Lives under build/shared (NOT the run
        dir), so reset_run's rmtree can't delete the counter it's tracking."""
        markers = self.shared_dir / "reset_markers"
        markers.mkdir(parents=True, exist_ok=True)
        return markers / f"{truncated_hash}.json"

    def bump_reset_marker(self, truncated_hash: str, signature: str) -> int:
        """Increment the consecutive-launch counter for a run, returning the new
        count. The counter resets to 1 whenever the checkpoint signature differs
        from the last launch (the previous run made progress = not a crash loop).
        """
        path = self._reset_marker_path(truncated_hash)
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                data = {}
        attempts = int(data.get("attempts", 0)) + 1 if data.get("checkpoint") == signature else 1
        try:
            path.write_text(json.dumps({"attempts": attempts, "checkpoint": signature}))
        except OSError:
            pass
        return attempts

    def clear_reset_marker(self, truncated_hash: str) -> None:
        """Drop a run's reset counter (after an auto-reset fires)."""
        try:
            self._reset_marker_path(truncated_hash).unlink()
        except OSError:
            pass

    def get_run_dir(self, truncated_hash: str) -> Path:
        """Get the directory path for a specific run.

        Args:
            truncated_hash: The truncated hash identifying the run

        Returns:
            Path to the run directory
        """
        return self.runs_dir / truncated_hash

    def setup_run(
        self,
        truncated_hash: str,
        full_command: str,
        args_hash: str,
        preserve: bool = False,
    ) -> Tuple[Path, bool]:
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
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}

        # Update config (preserve existing preserve flag if set)
        config.update(
            {
                "truncated_hash": truncated_hash,
                "full_hash": args_hash,
                "command": full_command,
                "last_updated": datetime.now().isoformat(),
                "preserve": config.get("preserve", False) or preserve,
            }
        )

        # If this is a new run, add creation time
        if not is_existing:
            config["created"] = datetime.now().isoformat()

        # Save config
        with open(config_path, "w") as f:
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
                with open(config_path, "r") as f:
                    config = json.load(f)
            else:
                # Basic info for runs without config
                config = {"truncated_hash": run_path.name, "preserve": False}

            # Add directory size
            size = sum(f.stat().st_size for f in run_path.rglob("*") if f.is_file())
            config["size_bytes"] = size
            config["size_human"] = self._format_bytes(size)

            # Check if this is the current run
            if self.current_link.exists():
                try:
                    current_target = self.current_link.resolve().name
                    config["is_current"] = current_target == run_path.name
                except:
                    config["is_current"] = False
            else:
                config["is_current"] = False

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
                with open(config_path, "r") as f:
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
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}

        config["preserve"] = preserve
        config["preserve_updated"] = datetime.now().isoformat()

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        status = "preserved" if preserve else "unpreserved"
        print(f"Run {truncated_hash} marked as {status}")
