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
    
    def fast_cuda_cleanup(self, timeout: float = 2.0):
        """Perform fast CUDA cleanup during shutdown.
        
        Args:
            timeout: Maximum time to wait for CUDA operations
        """
        if not torch.cuda.is_available():
            return
            
        print("[CUDA] Starting fast cleanup...")
        start_time = time.time()
        
        try:
            # 1. Stop all pending CUDA operations gracefully
            for device_id in range(torch.cuda.device_count()):
                try:
                    torch.cuda.set_device(device_id)
                    
                    # Check if we have time left
                    if time.time() - start_time > timeout:
                        print(f"[CUDA] Timeout reached, skipping device {device_id}")
                        break
                    
                    # Try to synchronize with a short timeout (non-blocking check first)
                    stream = torch.cuda.current_stream(device_id)
                    
                    # Use events for non-blocking check
                    event = torch.cuda.Event()
                    event.record(stream)
                    
                    # Wait with timeout
                    wait_time = min(0.5, timeout - (time.time() - start_time))
                    if wait_time > 0:
                        # This is a busy wait, but with a timeout
                        end_wait = time.time() + wait_time
                        while not event.query() and time.time() < end_wait:
                            time.sleep(0.01)  # Small sleep to avoid burning CPU
                    
                    if not event.query():
                        print(f"[CUDA] Device {device_id} operations still pending, continuing anyway")
                    
                except Exception as e:
                    # Ignore errors during cleanup
                    pass
            
            # 2. Clear cache to free memory
            try:
                torch.cuda.empty_cache()
                print("[CUDA] Cache cleared")
            except:
                pass
            
            # 3. Reset peak memory stats to avoid warnings
            try:
                for device_id in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(device_id)
                    torch.cuda.reset_accumulated_memory_stats(device_id)
            except:
                pass
                
        except Exception as e:
            print(f"[CUDA] Cleanup warning: {e}")
        
        elapsed = time.time() - start_time
        print(f"[CUDA] Cleanup completed in {elapsed:.2f}s")
    
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
                warnings.warn("Skipping full CUDA synchronization during shutdown", RuntimeWarning)
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