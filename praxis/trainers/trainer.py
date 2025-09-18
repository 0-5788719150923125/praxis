"""Framework-agnostic trainer implementation."""

from typing import Any, Dict, List, Optional, Union
import logging
from praxis.trainers.base import BaseTrainer, TrainerConfig


class Trainer:
    """Framework-agnostic trainer that delegates to specific implementations.

    This class provides a unified interface for training that works with
    different backend frameworks (Lightning, etc.).
    """

    def __init__(self, config: Optional[TrainerConfig] = None, **kwargs):
        """Initialize the trainer.

        Args:
            config: TrainerConfig with training settings
            **kwargs: Additional framework-specific arguments
        """
        self.config = config or TrainerConfig()
        self.backend = None
        self._initialize_backend(**kwargs)

    def _initialize_backend(self, **kwargs):
        """Initialize the backend trainer based on available frameworks."""
        # Try Lightning first
        try:
            from lightning.pytorch import Trainer as LightningTrainer

            # Convert config to Lightning trainer kwargs
            trainer_kwargs = self._config_to_lightning_kwargs(self.config)
            trainer_kwargs.update(kwargs)
            self.backend = LightningTrainer(**trainer_kwargs)
            self.framework = "lightning"
            return
        except ImportError:
            pass

        # If no backend available, raise error
        raise RuntimeError(
            "No training backend available. Please install PyTorch Lightning: "
            "pip install lightning"
        )

    def _config_to_lightning_kwargs(self, config: TrainerConfig) -> Dict[str, Any]:
        """Convert TrainerConfig to Lightning Trainer kwargs."""
        return {
            "max_epochs": config.max_epochs if config.max_epochs is not None else -1,
            "max_steps": config.max_steps if config.max_steps is not None else -1,
            "val_check_interval": config.val_check_interval,
            "log_every_n_steps": config.log_every_n_steps,
            "gradient_clip_val": config.gradient_clip_val,
            "accumulate_grad_batches": config.accumulate_grad_batches,
            "precision": config.precision,
            "devices": config.devices,
            "accelerator": config.accelerator,
            "strategy": config.strategy,
            "num_nodes": config.num_nodes,
            "enable_checkpointing": config.enable_checkpointing,
            "enable_progress_bar": config.enable_progress_bar,
            "enable_model_summary": config.enable_model_summary,
            "default_root_dir": config.default_root_dir,
            "fast_dev_run": config.fast_dev_run,
            "limit_train_batches": config.limit_train_batches,
            "limit_val_batches": config.limit_val_batches,
            "limit_test_batches": config.limit_test_batches,
            "overfit_batches": config.overfit_batches,
            "check_val_every_n_epoch": config.check_val_every_n_epoch,
            "profiler": config.profiler,
            "detect_anomaly": config.detect_anomaly,
            "barebones": config.barebones,
            "plugins": config.plugins,
            "sync_batchnorm": config.sync_batchnorm,
            "reload_dataloaders_every_n_epochs": config.reload_dataloaders_every_n_epochs,
            "use_distributed_sampler": config.use_distributed_sampler,
        }

    def fit(self, model: Any, datamodule: Any, **kwargs) -> None:
        """Train the model.

        Args:
            model: Model to train (BaseTrainingModule or Lightning module)
            datamodule: Data module (BaseDataModule or Lightning data module)
            **kwargs: Additional training arguments
        """
        if self.backend is None:
            raise RuntimeError("No backend initialized")

        return self.backend.fit(model, datamodule, **kwargs)

    def validate(self, model: Any, datamodule: Any, **kwargs) -> Any:
        """Validate the model.

        Args:
            model: Model to validate
            datamodule: Data module
            **kwargs: Additional validation arguments

        Returns:
            Validation metrics
        """
        if self.backend is None:
            raise RuntimeError("No backend initialized")

        return self.backend.validate(model, datamodule, **kwargs)

    def test(self, model: Any, datamodule: Any, **kwargs) -> Any:
        """Test the model.

        Args:
            model: Model to test
            datamodule: Data module
            **kwargs: Additional test arguments

        Returns:
            Test metrics
        """
        if self.backend is None:
            raise RuntimeError("No backend initialized")

        return self.backend.test(model, datamodule, **kwargs)

    def predict(self, model: Any, datamodule: Any, **kwargs) -> Any:
        """Generate predictions.

        Args:
            model: Model for predictions
            datamodule: Data module
            **kwargs: Additional prediction arguments

        Returns:
            Predictions
        """
        if self.backend is None:
            raise RuntimeError("No backend initialized")

        return self.backend.predict(model, datamodule, **kwargs)
