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
        # Extract metrics from trainer.callback_metrics
        metrics = {}

        for key, value in trainer.callback_metrics.items():
            if key in ['loss', 'learning_rate', 'num_tokens', 'softmax_collapse',
                      'rl_reward_mean', 'rl_reward_max', 'avg_step_time']:
                try:
                    # Convert tensor to float
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    pass

        if metrics:
            self.metrics_logger.log(step=trainer.global_step, **metrics)

    def on_validation_end(self, trainer, pl_module):
        """Log validation metrics."""
        metrics = {}

        for key, value in trainer.callback_metrics.items():
            if key.startswith('val_'):
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    pass

        if metrics:
            self.metrics_logger.log(step=trainer.global_step, **metrics)

    def on_train_end(self, trainer, pl_module):
        """Close logger on training end."""
        self.metrics_logger.close()
