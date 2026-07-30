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
        # Bound here, not only in on_fit_start: a signal arriving before the
        # fit starts (dataset setup can take minutes) otherwise raised
        # AttributeError inside the cleanup thread, skipping every remaining
        # step - dataloaders, CUDA, wandb, and the force-exit watchdog that is
        # the last line of defence against a hung shutdown.
        self.terminal_interface = None
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

    @staticmethod
    def _emit(message: str) -> None:
        """Print from signal context without ever raising.

        Python delivers a signal handler on the MAIN thread between bytecodes,
        so anything this raises surfaces as an exception at whatever line the
        main thread happened to be executing - a traceback rooted in unrelated
        code (a matmul, an optimizer scan) that the caller then has no way to
        recognise as "the user pressed Ctrl+C". A closed or detached stdout is
        enough to trigger it: printing to one raises ValueError("I/O operation
        on closed file"), and a clean cancellation gets reported as a crash.

        stdout is tried first (the dashboard captures it into the log panel),
        then the process's real stderr, then nothing at all. Silence is the
        correct last resort here - the shutdown itself still proceeds.
        """
        for stream in (sys.stdout, sys.__stderr__):
            try:
                if stream is None or getattr(stream, "closed", False):
                    continue
                stream.write(message)
                stream.flush()
                return
            except Exception:
                continue

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals with minimal operations in signal context.

        Total by construction: nothing here may propagate, because an exception
        escaping a signal handler is indistinguishable from a genuine crash in
        the code it interrupted. Each step is independently guarded so a failure
        in one (a dead stream, a half-torn-down trainer) cannot skip the rest.
        """
        try:
            with self.shutdown_lock:
                self.signal_count += 1
                count = self.signal_count
        except Exception:
            count = 1

        if count == 1:
            # Flag FIRST, announce second: the flag is what stops training and
            # what lets teardown tell a cancellation from a crash, so it must
            # not be downstream of an I/O call that can fail.
            try:
                self.cuda_manager.request_shutdown()
            except Exception:
                pass
            try:
                if self.trainer_ref is not None:
                    self.trainer_ref.should_stop = True
            except Exception:
                pass
            self._emit("\n🛑 Gracefully stopping training...\n")
            # Defer all cleanup to a thread outside signal context
            try:
                threading.Thread(target=self._deferred_cleanup, daemon=True).start()
            except Exception:
                pass
            # DON'T re-send the signal - just return and let Lightning handle
            # shutdown.
            return

        if count == 2:
            # Second signal: Still try to be gentle
            self._emit("\n⚠️  Forcing shutdown...\n")
            try:
                self._immediate_cleanup()
            except Exception:
                pass
            sys.exit(130)

        # Third+ signal: immediate hard exit
        self._emit("\n💀 Emergency exit!\n")
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

        # 1. Restore terminal immediately (most important).
        #
        # Tell the interface it is shutting down BEFORE nulling the dashboard.
        # Training keeps running for a few batches after the signal, and the
        # inference hook falls back to printing when it finds no dashboard - so
        # nulling it first dumps rolling-context text straight onto the restored
        # terminal, mixed into whatever the shutdown is reporting.
        terminal = getattr(self, "terminal_interface", None)
        if terminal is not None:
            try:
                terminal.begin_shutdown()
            except Exception:
                pass
            try:
                dashboard = getattr(terminal, "dashboard", None)
                if dashboard:
                    dashboard.stop()
                    dashboard.__exit__(None, None, None)
                    terminal.dashboard = None
            except Exception:
                pass

        # Always try to restore cursor visibility
        try:
            sys.stderr.write("\033[?25h")
            sys.stderr.flush()
        except:
            pass

        # 2. Stop dataloaders
        try:
            if self.datamodule_ref and isinstance(
                self.datamodule_ref, PraxisDataModule
            ):
                self.datamodule_ref.shutdown_dataloaders(timeout=0.5)
        except Exception:
            pass

        # 3. Quick CUDA cleanup
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
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
                self._emit(
                    "\n⏱️  Timeout waiting for graceful shutdown, forcing exit...\n"
                )
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
