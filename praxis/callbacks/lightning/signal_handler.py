"""Lightning callback for graceful signal handling."""

import os
import signal
import sys
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
        """Handle shutdown signals with minimal operations in signal context."""
        with self.shutdown_lock:
            self.signal_count += 1

            if self.signal_count == 1:
                # First signal: Minimal work in signal handler
                print(f"\n🛑 Gracefully stopping training...")

                # Set flags only - no complex operations in signal handler
                self.cuda_manager.request_shutdown()
                if self.trainer_ref is not None:
                    self.trainer_ref.should_stop = True

                # Defer all cleanup to a thread outside signal context
                threading.Thread(target=self._deferred_cleanup, daemon=True).start()

                # DON'T re-send the signal - just return and let Lightning handle shutdown
                return

            elif self.signal_count == 2:
                # Second signal: Still try to be gentle
                print(f"\n⚠️  Forcing shutdown...")
                # Perform immediate cleanup before exit
                self._immediate_cleanup()
                sys.exit(130)

            else:
                # Third+ signal: immediate hard exit
                print(f"\n💀 Emergency exit!")
                os._exit(130)

    def _immediate_cleanup(self):
        """Perform immediate critical cleanup before forced exit."""
        # Always try to restore cursor visibility
        try:
            sys.stderr.write("\033[?25h")
            sys.stderr.flush()
        except:
            pass

        # Stop dashboard if running
        if self.terminal_interface and hasattr(self.terminal_interface, "dashboard"):
            try:
                dashboard = self.terminal_interface.dashboard
                if dashboard:
                    dashboard.stop()
            except:
                pass

    def _deferred_cleanup(self):
        """Perform cleanup operations outside of signal handler context."""
        # This runs in a separate thread, avoiding signal handler restrictions

        # 1. Restore terminal immediately (most important)
        if self.terminal_interface and hasattr(self.terminal_interface, "dashboard"):
            try:
                dashboard = self.terminal_interface.dashboard
                if dashboard:
                    dashboard.stop()
                    dashboard.__exit__(None, None, None)
                    self.terminal_interface.dashboard = None
            except:
                pass

        # Always try to restore cursor visibility
        try:
            sys.stderr.write("\033[?25h")
            sys.stderr.flush()
        except:
            pass

        # 2. Stop dataloaders
        if self.datamodule_ref and isinstance(self.datamodule_ref, PraxisDataModule):
            try:
                self.datamodule_ref.shutdown_dataloaders(timeout=0.5)
            except:
                pass

        # 3. Quick CUDA cleanup
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass

        # 4. Signal wandb to finish
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish(quiet=True)
        except:
            pass

        # 5. Start a watchdog timer to force exit if Lightning doesn't stop
        def force_exit_watchdog():
            """Force exit if Lightning doesn't stop within timeout."""
            import time

            time.sleep(5.0)  # Give Lightning 5 seconds to stop gracefully
            if self.signal_count == 1:  # Only if still on first signal
                print("\n⏱️  Timeout waiting for graceful shutdown, forcing exit...")
                os._exit(130)

        watchdog = threading.Thread(target=force_exit_watchdog, daemon=True)
        watchdog.start()

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
        """Handle exceptions including KeyboardInterrupt for proper cleanup."""
        # Perform cleanup when Lightning catches an exception
        if isinstance(exception, KeyboardInterrupt):
            print("   Performing cleanup after interrupt...")
            # Run our deferred cleanup if it hasn't run yet
            if self.signal_count == 0:
                self._deferred_cleanup()

        # Always restore handlers
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
        if self.signal_count > 0:
            # Force Lightning to check should_stop immediately
            if not trainer.should_stop:
                trainer.should_stop = True
            # Ensure no validation will run after stopping
            trainer.limit_val_batches = 0
            trainer.check_val_every_n_epoch = float("inf")  # Never validate
            # Return -1 to signal Lightning to stop this batch
            return -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Check for shutdown after each batch."""
        if self.signal_count > 0:
            if not trainer.should_stop:
                trainer.should_stop = True
            # Force Lightning to check the flag immediately
            return -1

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Check for shutdown before optimizer step (most frequent check)."""
        if self.signal_count > 0:
            if not trainer.should_stop:
                trainer.should_stop = True
            # Skip this optimizer step by returning -1
            return -1

    def on_validation_start(self, trainer, pl_module):
        """Check if we should skip validation entirely."""
        if self.signal_count > 0:
            # Skip validation if shutting down
            print("   Skipping validation due to shutdown...")
            trainer.limit_val_batches = 0
            # Also set sanity check to 0
            trainer.num_sanity_val_steps = 0
