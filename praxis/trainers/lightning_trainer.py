"""PyTorch Lightning trainer implementation."""

from typing import Any, Dict, List, Optional, Union
import warnings

# Lightning imports - centralized here
from lightning.fabric.utilities.seed import reset_seed, seed_everything
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import (
    Callback,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities import disable_possible_user_warnings

from praxis.trainers.base import BaseTrainer, BaseCallback, BaseLogger, TrainerConfig


class LightningTrainerWrapper(BaseTrainer):
    """Wrapper for PyTorch Lightning Trainer."""
    
    def __init__(self, config: TrainerConfig, callbacks: Optional[List[Any]] = None, 
                 logger: Optional[Any] = None):
        """Initialize Lightning trainer with configuration."""
        self.config = config
        
        # Convert config to Lightning Trainer kwargs
        trainer_kwargs = self._config_to_trainer_kwargs(config)
        
        # Add callbacks and logger
        if callbacks:
            trainer_kwargs['callbacks'] = callbacks
        if logger:
            trainer_kwargs['logger'] = logger
            
        self.trainer = Trainer(**trainer_kwargs)
    
    def _config_to_trainer_kwargs(self, config: TrainerConfig) -> Dict[str, Any]:
        """Convert TrainerConfig to Lightning Trainer kwargs."""
        return {
            'max_epochs': config.max_epochs,
            'max_steps': config.max_steps,
            'val_check_interval': config.val_check_interval,
            'log_every_n_steps': config.log_every_n_steps,
            'gradient_clip_val': config.gradient_clip_val,
            'accumulate_grad_batches': config.accumulate_grad_batches,
            'precision': config.precision,
            'devices': config.devices,
            'accelerator': config.accelerator,
            'strategy': config.strategy,
            'num_nodes': config.num_nodes,
            'enable_checkpointing': config.enable_checkpointing,
            'enable_progress_bar': config.enable_progress_bar,
            'enable_model_summary': config.enable_model_summary,
            'default_root_dir': config.default_root_dir,
            'fast_dev_run': config.fast_dev_run,
            'limit_train_batches': config.limit_train_batches,
            'limit_val_batches': config.limit_val_batches,
            'limit_test_batches': config.limit_test_batches,
            'overfit_batches': config.overfit_batches,
            'check_val_every_n_epoch': config.check_val_every_n_epoch,
            'profiler': config.profiler,
            'detect_anomaly': config.detect_anomaly,
            'barebones': config.barebones,
            'plugins': config.plugins,
            'sync_batchnorm': config.sync_batchnorm,
            'reload_dataloaders_every_n_epochs': config.reload_dataloaders_every_n_epochs,
            'use_distributed_sampler': config.use_distributed_sampler,
        }
    
    def fit(self, model: Any, datamodule: Any, **kwargs) -> None:
        """Train the model using Lightning."""
        self.trainer.fit(model, datamodule, **kwargs)
    
    def validate(self, model: Any, datamodule: Any, **kwargs) -> Dict[str, float]:
        """Validate the model."""
        return self.trainer.validate(model, datamodule, **kwargs)
    
    def test(self, model: Any, datamodule: Any, **kwargs) -> Dict[str, float]:
        """Test the model."""
        return self.trainer.test(model, datamodule, **kwargs)
    
    def predict(self, model: Any, datamodule: Any, **kwargs) -> Any:
        """Generate predictions."""
        return self.trainer.predict(model, datamodule, **kwargs)


# Factory functions for Lightning components
def create_lightning_trainer(
    config: TrainerConfig,
    callbacks: Optional[List[Any]] = None,
    logger: Optional[Any] = None,
) -> LightningTrainerWrapper:
    """Create a Lightning trainer with the given configuration."""
    return LightningTrainerWrapper(config, callbacks, logger)


def create_csv_logger(
    save_dir: str = ".",
    name: str = "default",
    version: Optional[Union[int, str]] = None,
    prefix: str = "",
    flush_logs_every_n_steps: int = 100,
) -> CSVLogger:
    """Create a CSV logger for Lightning."""
    return CSVLogger(
        save_dir=save_dir,
        name=name,
        version=version,
        prefix=prefix,
        flush_logs_every_n_steps=flush_logs_every_n_steps,
    )


def create_model_checkpoint(
    dirpath: Optional[str] = None,
    filename: Optional[str] = None,
    monitor: Optional[str] = None,
    mode: str = "min",
    save_last: Optional[bool] = None,
    save_top_k: int = 1,
    save_weights_only: bool = False,
    every_n_train_steps: Optional[int] = None,
    train_time_interval: Optional[Any] = None,
    every_n_epochs: Optional[int] = None,
    save_on_train_epoch_end: Optional[bool] = None,
    enable: bool = True,
) -> ModelCheckpoint:
    """Create a model checkpoint callback for Lightning."""
    return ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_last=save_last,
        save_top_k=save_top_k,
        save_weights_only=save_weights_only,
        every_n_train_steps=every_n_train_steps,
        train_time_interval=train_time_interval,
        every_n_epochs=every_n_epochs,
        save_on_train_epoch_end=save_on_train_epoch_end,
        enable=enable,
    )


def create_progress_bar(
    refresh_rate: int = 1,
    leave: bool = False,
    theme: Optional[Dict[str, str]] = None,
    console_kwargs: Optional[Dict[str, Any]] = None,
) -> TQDMProgressBar:
    """Create a progress bar callback for Lightning."""
    return TQDMProgressBar(
        refresh_rate=refresh_rate,
        leave=leave,
        theme=theme,
        console_kwargs=console_kwargs,
    )


# Utility functions
def disable_warnings() -> None:
    """Disable Lightning-specific warnings."""
    disable_possible_user_warnings()


def seed_everything_lightning(seed: int, workers: bool = False) -> int:
    """Set seed for everything including Lightning components."""
    return seed_everything(seed, workers=workers)


def reset_seed_lightning() -> None:
    """Reset seed using Lightning utilities."""
    reset_seed()


# Re-export Lightning components for backward compatibility
__all__ = [
    'LightningModule',
    'Trainer', 
    'Callback',
    'ModelCheckpoint',
    'TQDMProgressBar',
    'CSVLogger',
    'LightningTrainerWrapper',
    'create_lightning_trainer',
    'create_csv_logger',
    'create_model_checkpoint',
    'create_progress_bar',
    'disable_warnings',
    'seed_everything_lightning',
    'reset_seed_lightning',
]