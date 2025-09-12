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
                # First signal: request graceful shutdown
                print(f"\n🛑 Shutdown requested, stopping training...")

                # Immediately stop dashboard in main thread (critical for terminal state)
                if self.terminal_interface and hasattr(
                    self.terminal_interface, "dashboard"
                ):
                    try:
                        dashboard = self.terminal_interface.dashboard
                        if dashboard:
                            dashboard.stop()
                    except:
                        pass

                # Signal CUDA manager that shutdown is requested
                self.cuda_manager.request_shutdown()

                # Start parallel shutdown operations
                def parallel_shutdown():
                    """Run shutdown operations in parallel."""
                    threads = []

                    # 1. Shutdown dashboard first (most important for terminal state)
                    if self.terminal_interface and hasattr(
                        self.terminal_interface, "dashboard"
                    ):

                        def shutdown_dashboard():
                            try:
                                dashboard = self.terminal_interface.dashboard
                                if dashboard:
                                    dashboard.stop()
                                    # Call the exit handler to restore terminal
                                    dashboard.__exit__(None, None, None)
                                    self.terminal_interface.dashboard = None
                            except Exception as e:
                                # Fallback: at least try to restore cursor
                                try:
                                    import sys

                                    sys.stderr.write("\033[?25h")
                                    sys.stderr.flush()
                                except:
                                    pass

                        threads.append(
                            threading.Thread(target=shutdown_dashboard, daemon=True)
                        )

                    # 2. DataLoader shutdown
                    if self.datamodule_ref and isinstance(
                        self.datamodule_ref, PraxisDataModule
                    ):

                        def shutdown_dataloaders():
                            try:
                                self.datamodule_ref.shutdown_dataloaders(timeout=2.0)
                            except Exception as e:
                                print(f"   Warning: DataLoader shutdown error: {e}")

                        threads.append(
                            threading.Thread(target=shutdown_dataloaders, daemon=True)
                        )

                    # 3. CUDA cleanup (if GPU is being used)
                    if torch.cuda.is_available() and torch.cuda.is_initialized():

                        def cuda_cleanup():
                            try:
                                self.cuda_manager.fast_cuda_cleanup(timeout=2.0)
                            except Exception as e:
                                print(f"   Warning: CUDA cleanup error: {e}")

                        threads.append(
                            threading.Thread(target=cuda_cleanup, daemon=True)
                        )

                    # Start all threads
                    for thread in threads:
                        thread.start()

                    # Wait for completion with timeout
                    for thread in threads:
                        thread.join(timeout=3.0)

                # Start parallel shutdown in background
                shutdown_thread = threading.Thread(
                    target=parallel_shutdown, daemon=True
                )
                shutdown_thread.start()

                # Check if we're in validation
                if self.trainer_ref is not None:
                    self.trainer_ref.should_stop = True

                    # Disable any future validation (using writable properties)
                    self.trainer_ref.limit_val_batches = 0
                    self.trainer_ref.num_sanity_val_steps = 0

                    # If in validation, try to stop it immediately
                    if hasattr(self.trainer_ref, "state") and hasattr(
                        self.trainer_ref.state, "stage"
                    ):
                        stage = (
                            str(self.trainer_ref.state.stage)
                            if self.trainer_ref.state.stage
                            else ""
                        )
                        if "validat" in stage.lower():
                            print(f"   Interrupting validation...")
                        else:
                            print(f"   Current stage: {stage}")

                print(f"   (Press Ctrl+C again to force immediate shutdown)")

            elif self.signal_count == 2:
                # Second signal: more aggressive
                print(f"\n⚠️  Force stopping... (Press Ctrl+C once more for hard exit)")

                if self.trainer_ref is not None:
                    # Really try to stop validation
                    self.trainer_ref.limit_val_batches = 0
                    self.trainer_ref.should_stop = True

                # Force DataLoader shutdown if not done
                if self.datamodule_ref and isinstance(
                    self.datamodule_ref, PraxisDataModule
                ):
                    try:
                        self.datamodule_ref.shutdown_dataloaders(timeout=1.0)
                    except:
                        pass

                # Raise KeyboardInterrupt to trigger cleanup
                raise KeyboardInterrupt

            else:
                # Third+ signal: immediate exit
                print(f"\n💀 Hard exit!")

                # Restore original handler and re-raise
                if self.original_sigint is not None:
                    signal.signal(signal.SIGINT, self.original_sigint)

                # Force immediate termination
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
