"""CUDA-aware shutdown utilities for faster, safer termination."""

import torch
import threading
import time
import warnings
from typing import Optional, List


class CUDAShutdownManager:
    """Manages CUDA operations during shutdown for faster termination."""

    def __init__(self):
        """Initialize the CUDA shutdown manager."""
        self._shutdown_requested = False
        self._shutdown_lock = threading.Lock()
        self._original_synchronize = None
        self._streams_to_sync: List[torch.cuda.Stream] = []

    def request_shutdown(self):
        """Signal that shutdown has been requested."""
        with self._shutdown_lock:
            self._shutdown_requested = True

    def is_shutting_down(self) -> bool:
        """Check if shutdown has been requested."""
        with self._shutdown_lock:
            return self._shutdown_requested

    def register_stream(self, stream: torch.cuda.Stream):
        """Register a CUDA stream for tracking during shutdown."""
        with self._shutdown_lock:
            if stream not in self._streams_to_sync:
                self._streams_to_sync.append(stream)

    def fast_cuda_cleanup(self, timeout: float = 0.5):
        """Perform fast CUDA cleanup during shutdown.

        Args:
            timeout: Maximum time to wait for CUDA operations (default 0.5s for fast shutdown)
        """
        if not torch.cuda.is_available():
            return

        # Skip verbose output for faster shutdown
        start_time = time.time()

        try:
            # Just clear cache immediately - most important operation
            torch.cuda.empty_cache()
            
            # Only try quick sync if we have time
            if timeout > 0.1:
                for device_id in range(torch.cuda.device_count()):
                    if time.time() - start_time > timeout:
                        break
                    try:
                        torch.cuda.set_device(device_id)
                        # Non-blocking check only
                        event = torch.cuda.Event()
                        event.record()
                        # Just check, don't wait
                        event.query()
                    except:
                        pass

        except:
            # Silently ignore any errors during cleanup
            pass

    def patch_synchronize_for_shutdown(self):
        """Patch torch.cuda.synchronize to be aware of shutdown.

        During shutdown, synchronize operations will timeout quickly
        instead of waiting indefinitely.
        """
        if self._original_synchronize is not None:
            return  # Already patched

        self._original_synchronize = torch.cuda.synchronize

        def shutdown_aware_synchronize(device=None):
            """Synchronize with shutdown awareness."""
            if self.is_shutting_down():
                # During shutdown, don't wait long
                warnings.warn(
                    "Skipping full CUDA synchronization during shutdown", RuntimeWarning
                )
                return
            else:
                # Normal operation
                return self._original_synchronize(device)

        torch.cuda.synchronize = shutdown_aware_synchronize

    def restore_synchronize(self):
        """Restore original torch.cuda.synchronize function."""
        if self._original_synchronize is not None:
            torch.cuda.synchronize = self._original_synchronize
            self._original_synchronize = None


# Global instance
_cuda_shutdown_manager = CUDAShutdownManager()


def get_cuda_shutdown_manager() -> CUDAShutdownManager:
    """Get the global CUDA shutdown manager."""
    return _cuda_shutdown_manager


def enable_fast_cuda_shutdown():
    """Enable fast CUDA shutdown mode."""
    manager = get_cuda_shutdown_manager()
    manager.patch_synchronize_for_shutdown()
    return manager


def disable_fast_cuda_shutdown():
    """Disable fast CUDA shutdown mode."""
    manager = get_cuda_shutdown_manager()
    manager.restore_synchronize()
    return manager
