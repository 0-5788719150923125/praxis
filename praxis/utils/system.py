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
        self._shutdown_lock = threading.Lock()
        self._cleanup_functions = []
        self._child_processes = weakref.WeakSet()
        self._original_sigint = None
        self._original_sigterm = None
        self._interrupt_count = 0
        self._last_interrupt_time = 0
        
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
        current_time = time.time()
        
        with self._shutdown_lock:
            # Track rapid interrupts
            if current_time - self._last_interrupt_time < 1.0:
                self._interrupt_count += 1
            else:
                self._interrupt_count = 1
            self._last_interrupt_time = current_time
            
            # Force immediate exit on third rapid interrupt or if already shutting down
            if self._interrupt_count >= 3 or (self._shutting_down and self._interrupt_count >= 2):
                print("\n⚠️  Force terminating...")
                # Minimal cleanup before forced exit
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except:
                    pass
                os._exit(exit_code)
            
            if self._shutting_down:
                # Already shutting down, just wait
                print("\n⏳ Shutdown in progress (press Ctrl+C again to force exit)...")
                return
            
            self._shutting_down = True
        
        print("\n🛑 Initiating graceful shutdown...")
        
        # Step 1: Stop accepting new work
        os.environ['PRAXIS_SHUTTING_DOWN'] = '1'
        
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
            if hasattr(torch, '_inductor') and hasattr(torch._inductor, 'async_compile'):
                if hasattr(torch._inductor.async_compile, 'shutdown_compile_workers'):
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
            if hasattr(torch.multiprocessing, '_clean_shutdown'):
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
    
    def install_signal_handlers(self):
        """Install signal handlers for graceful shutdown."""
        # Store any existing handlers
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        
        # Install our handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        print(f"\n📡 Received {signal_name}")
        self.initiate_shutdown()


# Global shutdown manager instance
shutdown_manager = ShutdownManager()


def sigint_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) gracefully."""
    shutdown_manager.initiate_shutdown()


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
    return os.environ.get('PRAXIS_SHUTTING_DOWN', '0') == '1'


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
