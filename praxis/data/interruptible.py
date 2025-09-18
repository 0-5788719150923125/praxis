"""Interruptible DataLoader for graceful shutdown."""

import signal
import threading
import time
from typing import Optional, Any
from torch.utils.data import DataLoader
import torch.multiprocessing as mp


class InterruptibleDataLoader(DataLoader):
    """DataLoader that can be gracefully interrupted and shutdown."""

    def __init__(self, *args, **kwargs):
        """Initialize with standard DataLoader parameters plus shutdown support."""
        # Force some parameters for better shutdown behavior
        kwargs["persistent_workers"] = False  # Ensure clean worker termination

        # Only set timeout if we have workers (timeout not allowed with num_workers=0)
        num_workers = kwargs.get("num_workers", 0)
        if num_workers > 0 and "timeout" not in kwargs:
            kwargs["timeout"] = 60  # Default timeout for worker operations

        super().__init__(*args, **kwargs)

        self._shutdown_requested = False
        self._shutdown_event = threading.Event()  # For fast signaling
        self._shutdown_lock = threading.Lock()
        self._iterator = None

    def shutdown(self, timeout: float = 0.5):
        """Request fast shutdown of the DataLoader.

        Args:
            timeout: Maximum time to wait for workers to shutdown (default 0.5s for fast)
        """
        with self._shutdown_lock:
            if self._shutdown_requested:
                return  # Already shutting down

            self._shutdown_requested = True
            self._shutdown_event.set()  # Signal shutdown

            # Skip verbose output for faster shutdown
            # If we have an active iterator, force terminate it quickly
            if self._iterator is not None and self.num_workers > 0:
                try:
                    # Try quick shutdown first
                    if hasattr(self._iterator, "_shutdown_workers"):
                        # Very short timeout for fast shutdown
                        self.timeout = 0.1
                        self._iterator._shutdown_workers()
                except:
                    # If graceful fails, force immediately
                    pass
                finally:
                    # Always force cleanup for speed
                    self._force_cleanup()
                    self._iterator = None

    def _force_cleanup(self):
        """Force cleanup of worker processes quickly."""
        # Only relevant if we have workers
        if self.num_workers == 0:
            return

        try:
            if self._iterator is not None and hasattr(self._iterator, "_workers"):
                # Just terminate all workers immediately - no waiting
                for w in self._iterator._workers:
                    try:
                        if w.is_alive():
                            w.terminate()
                    except:
                        pass
        except:
            # Silently ignore errors for fast shutdown
            pass

    def __iter__(self):
        """Create iterator with shutdown awareness."""
        if self._shutdown_requested:
            # Don't start new iteration if shutdown requested
            return iter([])

        # Create the iterator and store reference
        self._iterator = super().__iter__()

        # Wrap the iterator to check for shutdown
        return self._shutdown_aware_iterator(self._iterator)

    def _shutdown_aware_iterator(self, iterator):
        """Wrap iterator to check for shutdown signals."""
        try:
            while not self._shutdown_requested:
                try:
                    # Try to get next batch with a small timeout
                    yield next(iterator)
                except StopIteration:
                    break
                except Exception as e:
                    if self._shutdown_requested:
                        # Expected during shutdown
                        break
                    raise e
        finally:
            # Ensure cleanup on iterator exit
            if self._shutdown_requested and self._iterator is not None:
                self._iterator = None

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            if not self._shutdown_requested:
                self.shutdown(timeout=2.0)
        except:
            pass  # Ignore errors during deletion


class DataLoaderManager:
    """Manager for tracking and shutting down multiple DataLoaders."""

    def __init__(self):
        """Initialize the manager."""
        self._dataloaders = []
        self._lock = threading.Lock()

    def register(self, dataloader: InterruptibleDataLoader):
        """Register a DataLoader for management."""
        with self._lock:
            self._dataloaders.append(dataloader)

    def shutdown_all(self, timeout: float = 0.5):
        """Shutdown all registered DataLoaders quickly."""
        with self._lock:
            if not self._dataloaders:
                return

            # Skip verbose output for faster shutdown
            # Signal all dataloaders to stop immediately
            for dl in self._dataloaders:
                try:
                    # Set shutdown flag directly for speed
                    dl._shutdown_requested = True
                    if hasattr(dl, '_shutdown_event'):
                        dl._shutdown_event.set()
                    # Quick force cleanup if there are workers
                    if dl.num_workers > 0:
                        dl._force_cleanup()
                except:
                    pass

            # Clear the list immediately - don't wait
            self._dataloaders.clear()
