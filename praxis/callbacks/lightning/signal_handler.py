"""Lightning callback for graceful signal handling."""

import signal
import threading
import torch
from lightning.pytorch.callbacks import Callback
from praxis.data.datamodule import PraxisDataModule
from praxis.utils.cuda_shutdown import get_cuda_shutdown_manager


class SignalHandlerCallback(Callback):
    """Handle signals gracefully by setting trainer.should_stop."""

    def __init__(self):
        """Initialize signal handler callback."""
        super().__init__()
        self.original_sigint = None
        self.original_sigterm = None
        self.signal_count = 0
        self.trainer_ref = None
        self.datamodule_ref = None
        self.shutdown_lock = threading.Lock()
        self.cuda_manager = get_cuda_shutdown_manager()

    def on_fit_start(self, trainer, pl_module):
        """Install signal handlers when training starts."""
        self.trainer_ref = trainer

        # Try to get datamodule reference if available
        if hasattr(trainer, "datamodule"):
            self.datamodule_ref = trainer.datamodule

        # Find TerminalInterface callback to access dashboard
        self.terminal_interface = None
        if hasattr(trainer, "callbacks"):
            for callback in trainer.callbacks:
                if callback.__class__.__name__ == "TerminalInterface":
                    self.terminal_interface = callback
                    break

        # Enable fast CUDA shutdown mode
        self.cuda_manager.patch_synchronize_for_shutdown()

        # Save original handlers
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self.original_sigterm = signal.getsignal(signal.SIGTERM)

        # Install our handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        with self.shutdown_lock:
            self.signal_count += 1

            if self.signal_count == 1:
                # First signal: Fast but safe shutdown
                print(f"\n🛑 Stopping immediately...")

                # Signal CUDA manager for fast shutdown
                self.cuda_manager.request_shutdown()

                # Stop trainer immediately
                if self.trainer_ref is not None:
                    self.trainer_ref.should_stop = True
                    self.trainer_ref.limit_val_batches = 0
                    self.trainer_ref.num_sanity_val_steps = 0

                # Restore terminal immediately (most important)
                if self.terminal_interface and hasattr(
                    self.terminal_interface, "dashboard"
                ):
                    try:
                        dashboard = self.terminal_interface.dashboard
                        if dashboard:
                            dashboard.stop()
                            dashboard.__exit__(None, None, None)
                            self.terminal_interface.dashboard = None
                    except:
                        # Fallback: restore cursor visibility
                        try:
                            import sys
                            sys.stderr.write("\033[?25h")
                            sys.stderr.flush()
                        except:
                            pass

                # Quick cleanup in background thread (non-blocking)
                def quick_cleanup():
                    """Minimal cleanup operations."""
                    # 1. Stop dataloaders (fast timeout)
                    if self.datamodule_ref and isinstance(
                        self.datamodule_ref, PraxisDataModule
                    ):
                        try:
                            self.datamodule_ref.shutdown_dataloaders(timeout=0.5)
                        except:
                            pass

                    # 2. Quick CUDA cleanup
                    if torch.cuda.is_available():
                        try:
                            # Just empty cache, don't wait for sync
                            torch.cuda.empty_cache()
                        except:
                            pass

                # Start cleanup but don't wait
                threading.Thread(target=quick_cleanup, daemon=True).start()

                # Immediately raise KeyboardInterrupt to stop Lightning
                raise KeyboardInterrupt

            else:
                # Second+ signal: immediate hard exit
                print(f"\n💀 Force exit!")
                
                # Restore terminal before exit
                try:
                    import sys
                    sys.stderr.write("\033[?25h")  # Show cursor
                    sys.stderr.flush()
                except:
                    pass
                
                # Immediate termination
                import os
                os._exit(130)

    def on_train_end(self, trainer, pl_module):
        """Called when training ends, before final validation."""
        if self.signal_count > 0:
            # Skip final validation if we're shutting down
            print("   Skipping final validation due to shutdown...")
            trainer.limit_val_batches = 0  # This is writable
            trainer.num_sanity_val_steps = 0  # This is also writable

    def on_fit_end(self, trainer, pl_module):
        """Restore original signal handlers."""
        # Restore CUDA synchronize function
        self.cuda_manager.restore_synchronize()

        # Restore signal handlers
        if self.original_sigint is not None:
            signal.signal(signal.SIGINT, self.original_sigint)
        if self.original_sigterm is not None:
            signal.signal(signal.SIGTERM, self.original_sigterm)

    def on_exception(self, trainer, pl_module, exception):
        """Restore handlers on exception."""
        self.on_fit_end(trainer, pl_module)

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        """Check if we should stop validation."""
        if self.signal_count > 0:
            # Skip remaining validation batches immediately
            trainer.limit_val_batches = batch_idx  # Stop after current batch
            trainer.should_stop = True

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Check if we should stop training."""
        if self.signal_count > 0 and trainer.should_stop:
            # Ensure no validation will run after stopping
            trainer.limit_val_batches = 0
            trainer.check_val_every_n_epoch = float("inf")  # Never validate
            return  # Lightning will handle stopping

    def on_validation_start(self, trainer, pl_module):
        """Check if we should skip validation entirely."""
        if self.signal_count > 0:
            # Skip validation if shutting down
            print("   Skipping validation due to shutdown...")
            trainer.limit_val_batches = 0
            # Also set sanity check to 0
            trainer.num_sanity_val_steps = 0
