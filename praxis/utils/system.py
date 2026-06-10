"""System utilities for process management and updates."""

import atexit
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import weakref
from glob import glob


def configure_multiprocessing():
    """Force the ``spawn`` multiprocessing start method.

    Fork-based workers deadlock when CUDA is initialized before the fork;
    ``spawn`` avoids this by re-importing from scratch in each worker. Must
    run before any CUDA work or DataLoader worker is created, so main.py
    calls this first.
    """
    import torch.multiprocessing as mp

    # Silence the resource_tracker's "leaked semaphore" warning at shutdown. It
    # runs in its own process and reads filters from PYTHONWARNINGS at spawn, so
    # an in-process filter can't reach it; append (don't clobber) an existing var.
    _filter = "ignore::UserWarning:multiprocessing.resource_tracker"
    existing = os.environ.get("PYTHONWARNINGS")
    os.environ["PYTHONWARNINGS"] = f"{existing},{_filter}" if existing else _filter

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set


def configure_cuda_allocator():
    """Default the CUDA caching allocator to expandable segments.

    Variable-shape workloads (Titans segmentation, byte-latent patches) fragment
    the default allocator: reserved VRAM balloons far past the live tensors.
    Expandable segments grow/shrink in place instead of reserving a fixed block
    per shape. The allocator reads this only at CUDA init, so main.py sets it
    first; an explicit PYTORCH_CUDA_ALLOC_CONF in the environment still wins.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class ShutdownManager:
    """Centralized shutdown manager for graceful termination."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._shutting_down = False
        self._shutdown_requested = False  # Flag set by signal handler
        self._shutdown_lock = threading.Lock()
        self._cleanup_functions = []
        self._child_processes = weakref.WeakSet()
        self._original_sigint = None
        self._original_sigterm = None
        self._interrupt_count = 0
        self._shutdown_thread = None

        # Register atexit handler for normal program termination
        atexit.register(self._cleanup_at_exit)

    def register_cleanup(self, func, priority=50):
        """Register a cleanup function with priority (lower = earlier execution)."""
        with self._shutdown_lock:
            self._cleanup_functions.append((priority, func))
            self._cleanup_functions.sort(key=lambda x: x[0])

    def register_process(self, process):
        """Register a child process for tracking."""
        self._child_processes.add(process)

    def initiate_shutdown(self, exit_code=0, force=False):
        """Initiate graceful shutdown sequence."""
        with self._shutdown_lock:
            if self._shutting_down:
                return
            self._shutting_down = True

        # Step 1: Stop accepting new work
        os.environ["PRAXIS_SHUTTING_DOWN"] = "1"

        # Step 2: Send termination signals to child processes
        for proc in list(self._child_processes):
            try:
                if proc.is_alive():
                    # Send SIGTERM to allow graceful shutdown
                    proc.terminate()
            except:
                pass

        # Step 3: Shutdown torch compile workers first
        try:
            import torch

            if hasattr(torch, "_inductor") and hasattr(
                torch._inductor, "async_compile"
            ):
                if hasattr(torch._inductor.async_compile, "shutdown_compile_workers"):
                    # Forcefully clear the compile workers without waiting
                    torch._inductor.async_compile._compile_worker_pool = None
        except:
            pass

        # Step 4: Execute registered cleanup functions
        for priority, func in self._cleanup_functions:
            try:
                func()
            except Exception as e:
                print(f"  Warning: Cleanup function failed: {e}")

        # Step 5: Wait briefly for child processes to terminate
        wait_start = time.time()
        max_wait = 2.0  # Maximum 2 seconds wait

        while time.time() - wait_start < max_wait:
            alive_procs = [p for p in self._child_processes if p.is_alive()]
            if not alive_procs:
                break
            time.sleep(0.1)

        # Step 6: Force kill any remaining processes
        for proc in list(self._child_processes):
            try:
                if proc.is_alive():
                    proc.kill()  # Force kill
            except:
                pass

        # Step 7: PyTorch-specific cleanup
        try:
            import torch

            # Synchronize CUDA devices
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        with torch.cuda.device(i):
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                    except:
                        pass

            # Clean up distributed training
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.destroy_process_group()
                except:
                    pass

            # Clear any multiprocessing queues
            if hasattr(torch.multiprocessing, "_clean_shutdown"):
                torch.multiprocessing._clean_shutdown()
        except ImportError:
            pass
        except Exception:
            pass

        # Step 8: Flush output streams
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except:
            pass

        print("✓ Shutdown complete")

        # Step 9: Exit cleanly
        sys.exit(exit_code)

    def _cleanup_at_exit(self):
        """Cleanup function called at normal program exit."""
        if not self._shutting_down:
            # Normal exit, just ensure streams are flushed
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except:
                pass

    def _signal_handler(self, signum, frame):
        """Deprecated - Let Lightning handle signals."""
        # DO NOT handle signals here
        pass

    def _delayed_shutdown(self):
        """Deprecated - Let Lightning handle shutdown."""
        # DO NOT perform delayed shutdown
        pass


# Global shutdown manager instance
shutdown_manager = ShutdownManager()


def sigint_handler(signum, frame):
    """Deprecated - Let Lightning handle signals."""
    # DO NOT handle signals here
    pass


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

    if not os.path.isdir(ckpt_dir):
        return None

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


def checkpoint_readable(path):
    """True if the checkpoint's zip central directory is intact. A save
    killed mid-write leaves a truncated archive that torch.load only
    rejects after the trainer is fully constructed."""
    import zipfile

    try:
        with zipfile.ZipFile(path):
            return True
    except (zipfile.BadZipFile, OSError):
        return False


def resolve_resume_checkpoint(cache_dir, reset=False):
    """Find a checkpoint to resume from, or None to start fresh.

    Prefers Lightning's ``last.ckpt`` symlink, then batch checkpoints
    newest-first, then a Mono-Forward ``mono_forward.pt`` (saved to a
    different path than Lightning). Corrupt candidates are skipped with
    a warning instead of crashing the launch. Honors --reset and the
    force_reset environment feature.
    """
    from praxis.environments import EnvironmentFeatures

    if reset or EnvironmentFeatures.is_enabled("force_reset"):
        return None

    ckpt_dir = os.path.join(cache_dir, "model")
    candidates = [os.path.join(ckpt_dir, "last.ckpt")]
    if os.path.isdir(ckpt_dir):
        batches = []
        for f in os.listdir(ckpt_dir):
            match = re.search(r"batch=(\d+)\.0\.ckpt", f)
            if match:
                batches.append((int(match.group(1)), f))
        candidates += [
            os.path.join(ckpt_dir, f) for _, f in sorted(batches, reverse=True)
        ]
    candidates.append(os.path.join(cache_dir, "mono_forward.pt"))

    seen = set()
    for path in candidates:
        real = os.path.realpath(path)
        if real in seen or not os.path.exists(path):
            continue
        seen.add(real)
        if not checkpoint_readable(path):
            print(f"[WARN] skipping corrupt checkpoint: {path}")
            continue
        print(f"resuming from checkpoint: {path}")
        return path

    return None


def _materialize_skipped_lazy_modules(model, device):
    """Probe any LazyModule left uninitialized after the main dummy forward.

    Per-depth ``nn.ModuleList`` branches (e.g. ``ArcGLU.act``) only see a
    forward at ``current_depth=0``, so siblings keep their
    ``UninitializedParameter`` and break any subsequent ``numel()`` call.
    For each such list we borrow the feature dim from an already-
    initialized sibling and run each skipped entry once.
    """
    import torch
    from torch.nn import ModuleList
    from torch.nn.modules.lazy import LazyModuleMixin

    for module in model.modules():
        if not isinstance(module, ModuleList):
            continue
        lazy_entries = [m for m in module if isinstance(m, LazyModuleMixin)]
        if not lazy_entries:
            continue
        initialized = next(
            (m for m in lazy_entries if not m.has_uninitialized_params()),
            None,
        )
        if initialized is None:
            continue  # nothing ran; leave it for the error path to surface

        feat_dim = None
        for p in initialized.parameters(recurse=False):
            if p.dim() >= 1:
                feat_dim = p.shape[-1]
                break
        if feat_dim is None:
            continue

        dummy = torch.zeros(1, feat_dim, device=device)
        with torch.no_grad():
            for entry in lazy_entries:
                if entry.has_uninitialized_params():
                    entry(dummy)


def initialize_lazy_modules(model, device):
    """Initialize lazy modules in a model by doing a dummy forward pass.

    This function is optimized to be interruptible during bootstrap.
    """
    import signal

    import torch

    # Check if we should skip initialization due to shutdown
    def check_interrupt():
        """Check if we've received an interrupt signal."""
        # This will be caught by the signal handler if Ctrl+C is pressed
        pass

    try:
        print("[INIT] Moving model to device...")
        check_interrupt()

        # Move model to device in chunks to be more interruptible
        model = model.to(device)

        # Use smaller dummy batch for faster initialization
        batch_size = 1  # Reduced from 2
        seq_length = 32  # Reduced from 64

        print("[INIT] Initializing lazy modules...")
        check_interrupt()

        # Create dummy batch for initialization
        dummy_input = torch.ones((batch_size, seq_length), dtype=torch.long).to(device)

        # Standard autoregressive shifting for all models
        dummy_labels = dummy_input[..., 1:].contiguous()

        # Do a dummy forward pass to initialize lazy parameters
        model.train()

        # Use no_grad to speed up initialization (we don't need gradients here)
        with torch.no_grad():
            outputs = model(input_ids=dummy_input, labels=dummy_labels)

        # Second pass: any LazyModule living inside a ModuleList whose
        # siblings were skipped by the main forward (e.g. ArcGLU's
        # per-depth activation list, where only self.act[current_depth]
        # runs) still has UninitializedParameter. Probe each skipped
        # entry directly, borrowing its initialized sibling's feature
        # dim so the dummy tensor lines up.
        _materialize_skipped_lazy_modules(model, device)

        # Clear any cached memory immediately
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[INIT] Model initialization complete")

    except KeyboardInterrupt:
        print("\n[INIT] Initialization interrupted")
        # Clean up partial initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    return model


def register_cleanup_function(func, priority=50):
    """Register a cleanup function with the shutdown manager.

    Args:
        func: Function to call during shutdown
        priority: Lower numbers execute first (default 50)
    """
    shutdown_manager.register_cleanup(func, priority)


def register_child_process(process):
    """Register a child process with the shutdown manager.

    Args:
        process: multiprocessing.Process or similar object
    """
    shutdown_manager.register_process(process)


def is_shutting_down():
    """Check if the system is currently shutting down."""
    # Check both the environment variable and the shutdown manager's flag
    return (
        os.environ.get("PRAXIS_SHUTTING_DOWN", "0") == "1"
        or shutdown_manager._shutdown_requested
    )


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
    full_repr = repr(model)
    repr_lines = full_repr.splitlines()

    # Filter out wrapper-related lines that might still be present
    # Start from the first line that contains "PraxisForCausalLM"
    start_idx = 0
    for i, line in enumerate(repr_lines):
        if "PraxisForCausalLM" in line:
            start_idx = i
            break

    # Find the matching closing parenthesis for PraxisForCausalLM
    # Count parenthesis depth starting from PraxisForCausalLM line
    end_idx = len(repr_lines)
    paren_depth = 0
    for i in range(start_idx, len(repr_lines)):
        line = repr_lines[i]
        # Count opening and closing parentheses
        paren_depth += line.count("(") - line.count(")")
        # When we return to depth 0, we've found the matching closing paren
        if i > start_idx and paren_depth == 0:
            end_idx = i + 1
            break

    # Take lines from PraxisForCausalLM to its matching closing parenthesis
    plan = repr_lines[start_idx:end_idx]

    launch_duration = random.uniform(6.7, 7.3)
    acceleration_curve = random.uniform(3.5, 4.5)
    start_time = time.time()

    time.sleep(max(0, random.gauss(1.0, 3.0)))

    # Print opening backticks
    for i, line in enumerate(plan):
        print(f"{line}")
        progress = i / len(plan)
        scale_factor = launch_duration * (acceleration_curve + 1) / len(plan)
        delay = scale_factor * (progress**acceleration_curve)
        time.sleep(delay)

    print(f"[HASH] {truncated_hash}")
    time.sleep(2)
    elapsed_time = time.time() - start_time
    print(f"[RATE] {elapsed_time:.3f}s")
    time.sleep(1)


def graceful_shutdown(api_server, exit_code=0, reason="training complete"):
    """Explicit teardown before Python finalization to avoid GIL races.

    Background daemon threads (Flask/Werkzeug, dataloader workers,
    integration clients) can still be in C-extension code when the main
    thread returns. If one calls ``PyGILState_Release`` on an interpreter
    that's already finalizing, the process dies with a fatal error even
    after a clean run. We stop each known resource with a short per-
    component timeout, then bypass finalization via ``os._exit``. Metrics
    DBs and checkpoints are already flushed by ``trainer.fit`` by now.
    """
    from praxis.cli import integration_loader

    print(f"[SHUTDOWN] {reason}; stopping background services...")

    def _stop_with_timeout(name, fn, timeout):
        try:
            t = threading.Thread(target=fn, daemon=True, name=f"shutdown_{name}")
            t.start()
            t.join(timeout=timeout)
            if t.is_alive():
                print(
                    f"[SHUTDOWN] {name} stop timed out after {timeout:.0f}s; "
                    "continuing teardown"
                )
        except Exception as exc:
            print(f"[SHUTDOWN] {name} stop raised: {exc!r}")

    if api_server is not None:
        _stop_with_timeout("api_server", api_server.stop, 5.0)

    try:
        _stop_with_timeout("integrations", integration_loader.run_cleanup_hooks, 5.0)
    except Exception as exc:
        print(f"[SHUTDOWN] integration cleanup failed to dispatch: {exc!r}")

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    # os._exit skips atexit handlers, finalizers, and the module-teardown
    # pass where the GIL races happen. Buffers are already flushed above.
    os._exit(exit_code)


def update_license_timestamp():
    """Update the LICENSE file's copyright line with year progress (0-1)."""
    from datetime import datetime

    now = datetime.now()
    year_start = datetime(now.year, 1, 1)
    year_end = datetime(now.year + 1, 1, 1)
    year_progress = (now - year_start).total_seconds() / (
        year_end - year_start
    ).total_seconds()

    with open("LICENSE", "r") as f:
        lines = f.readlines()

    if len(lines) >= 3 and "Copyright (c)" in lines[2]:
        fraction = str(year_progress).split(".", 1)[1]
        lines[2] = f"Copyright (c) {now.year}.{fraction}\n"
        with open("LICENSE", "w") as f:
            f.writelines(lines)
