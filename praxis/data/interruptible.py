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
        self._shutdown_lock = threading.Lock()
        self._iterator = None

    def shutdown(self, timeout: float = 5.0):
        """Request graceful shutdown of the DataLoader.

        Args:
            timeout: Maximum time to wait for workers to shutdown
        """
        with self._shutdown_lock:
            if self._shutdown_requested:
                return  # Already shutting down

            self._shutdown_requested = True

            # Only log if we actually have workers
            if self.num_workers > 0:
                print("   DataLoader: Initiating graceful shutdown...")

            # If we have an active iterator, try to shut it down
            if self._iterator is not None:
                try:
                    # Only try to shutdown workers if they exist
                    if (
                        hasattr(self._iterator, "_shutdown_workers")
                        and self.num_workers > 0
                    ):
                        # Save original timeout if it exists
                        original_timeout = getattr(self, "timeout", None)

                        # Set a timeout for worker shutdown
                        if original_timeout is not None:
                            self.timeout = min(timeout, 5.0)

                        # Call shutdown on the iterator
                        self._iterator._shutdown_workers()

                        # Restore original timeout
                        if original_timeout is not None:
                            self.timeout = original_timeout

                    # Clear the iterator reference
                    self._iterator = None

                    if self.num_workers > 0:
                        print("   DataLoader: Workers shutdown completed")

                except Exception as e:
                    if self.num_workers > 0:
                        print(f"   DataLoader: Warning during shutdown: {e}")
                    # Force cleanup if graceful fails
                    self._force_cleanup()

    def _force_cleanup(self):
        """Force cleanup of worker processes if graceful shutdown fails."""
        # Only relevant if we have workers
        if self.num_workers == 0:
            return

        try:
            if self._iterator is not None and hasattr(self._iterator, "_workers"):
                # Try to terminate worker processes
                for w in self._iterator._workers:
                    if w.is_alive():
                        w.terminate()
                        # Give it a moment to terminate
                        w.join(timeout=0.5)
                        if w.is_alive():
                            # If still alive, force kill (Linux only)
                            try:
                                import os

                                os.kill(w.pid, signal.SIGKILL)
                            except:
                                pass

                print("   DataLoader: Forced worker termination completed")
        except Exception as e:
            print(f"   DataLoader: Error during force cleanup: {e}")

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

    def shutdown_all(self, timeout: float = 5.0):
        """Shutdown all registered DataLoaders."""
        with self._lock:
            if not self._dataloaders:
                return

            # Check if any dataloader actually has workers
            has_workers = any(dl.num_workers > 0 for dl in self._dataloaders)

            if has_workers:
                print(
                    f"DataLoaderManager: Shutting down {len(self._dataloaders)} dataloaders..."
                )

            # Start shutdown for all dataloaders in parallel
            threads = []
            for dl in self._dataloaders:
                thread = threading.Thread(target=dl.shutdown, args=(timeout,))
                thread.start()
                threads.append(thread)

            # Wait for all shutdowns to complete
            for thread in threads:
                thread.join(timeout=timeout)

            # Clear the list
            self._dataloaders.clear()

            if has_workers:
                print("DataLoaderManager: All dataloaders shutdown complete")
