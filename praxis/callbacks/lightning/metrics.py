"""Lightning callback to integrate with MetricsLogger."""

from lightning.pytorch.callbacks import Callback
from praxis.logging.metrics_logger import MetricsLogger


class MetricsLoggerCallback(Callback):
    """PyTorch Lightning callback that logs metrics to MetricsLogger.

    This callback automatically extracts metrics from Lightning's logging
    and writes them to the MetricsLogger for web visualization.

    Args:
        run_dir: Directory for the current run (e.g., "build/runs/83492c812")
    """

    def __init__(self, run_dir: str):
        """Initialize callback.

        Args:
            run_dir: Directory for the current run
        """
        super().__init__()
        self.metrics_logger = MetricsLogger(run_dir)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log metrics after each training batch."""
        # Extract ALL metrics from trainer.callback_metrics
        metrics = {}

        for key, value in trainer.callback_metrics.items():
            # Skip 'step' as we pass it explicitly
            if key == "step":
                continue
            try:
                # Convert tensor to float
                metrics[key] = float(value)
            except (TypeError, ValueError):
                # Skip non-numeric values
                pass

        if metrics:
            self.metrics_logger.log(step=trainer.global_step, **metrics)

    def on_validation_end(self, trainer, pl_module):
        """Log ALL metrics at validation time.

        This ensures validation metrics don't overwrite training metrics when
        the API deduplicates entries by step, and allows any metric to flow through.
        """
        metrics = {}

        # Log ALL metrics without filtering
        for key, value in trainer.callback_metrics.items():
            # Skip 'step' as we pass it explicitly
            if key == "step":
                continue
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                # Skip non-numeric values
                pass

        if metrics:
            self.metrics_logger.log(step=trainer.global_step, **metrics)

    def on_train_end(self, trainer, pl_module):
        """Close logger on training end."""
        self.metrics_logger.close()
