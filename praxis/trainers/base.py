"""Base trainer interface for framework-agnostic training."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""

    @abstractmethod
    def fit(self, model: Any, datamodule: Any, **kwargs) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def validate(self, model: Any, datamodule: Any, **kwargs) -> Dict[str, float]:
        """Validate the model."""
        pass

    @abstractmethod
    def test(self, model: Any, datamodule: Any, **kwargs) -> Dict[str, float]:
        """Test the model."""
        pass

    @abstractmethod
    def predict(self, model: Any, datamodule: Any, **kwargs) -> Any:
        """Generate predictions."""
        pass


class BaseCallback(ABC):
    """Abstract base class for training callbacks."""

    @abstractmethod
    def on_train_start(self, trainer: BaseTrainer, model: Any) -> None:
        """Called when training starts."""
        pass

    @abstractmethod
    def on_train_end(self, trainer: BaseTrainer, model: Any) -> None:
        """Called when training ends."""
        pass

    @abstractmethod
    def on_epoch_start(self, trainer: BaseTrainer, model: Any) -> None:
        """Called at the start of each epoch."""
        pass

    @abstractmethod
    def on_epoch_end(self, trainer: BaseTrainer, model: Any) -> None:
        """Called at the end of each epoch."""
        pass

    @abstractmethod
    def on_batch_start(
        self, trainer: BaseTrainer, model: Any, batch: Any, batch_idx: int
    ) -> None:
        """Called before processing a batch."""
        pass

    @abstractmethod
    def on_batch_end(
        self, trainer: BaseTrainer, model: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Called after processing a batch."""
        pass


class BaseLogger(ABC):
    """Abstract base class for training loggers."""

    @abstractmethod
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics."""
        pass

    @abstractmethod
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Save any pending logs."""
        pass

    @abstractmethod
    def finalize(self, status: str = "success") -> None:
        """Finalize logging."""
        pass


class TrainerConfig:
    """Configuration for trainers."""

    def __init__(
        self,
        max_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        val_check_interval: Optional[Union[int, float]] = None,
        log_every_n_steps: int = 50,
        gradient_clip_val: Optional[float] = None,
        accumulate_grad_batches: int = 1,
        precision: str = "32-true",
        devices: Union[int, List[int], str] = 1,
        accelerator: str = "auto",
        strategy: str = "auto",
        num_nodes: int = 1,
        enable_checkpointing: bool = True,
        enable_progress_bar: bool = True,
        enable_model_summary: bool = True,
        default_root_dir: Optional[str] = None,
        fast_dev_run: Union[bool, int] = False,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        overfit_batches: Union[int, float] = 0.0,
        check_val_every_n_epoch: int = 1,
        profiler: Optional[str] = None,
        detect_anomaly: bool = False,
        barebones: bool = False,
        plugins: Optional[List[Any]] = None,
        sync_batchnorm: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        use_distributed_sampler: bool = True,
    ):
        """Initialize trainer configuration."""
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.val_check_interval = val_check_interval
        self.log_every_n_steps = log_every_n_steps
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.precision = precision
        self.devices = devices
        self.accelerator = accelerator
        self.strategy = strategy
        self.num_nodes = num_nodes
        self.enable_checkpointing = enable_checkpointing
        self.enable_progress_bar = enable_progress_bar
        self.enable_model_summary = enable_model_summary
        self.default_root_dir = default_root_dir
        self.fast_dev_run = fast_dev_run
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.limit_test_batches = limit_test_batches
        self.overfit_batches = overfit_batches
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.profiler = profiler
        self.detect_anomaly = detect_anomaly
        self.barebones = barebones
        self.plugins = plugins or []
        self.sync_batchnorm = sync_batchnorm
        self.reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs
        self.use_distributed_sampler = use_distributed_sampler

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


# Seed utilities that are framework-agnostic
def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def reset_random_state(seed: Optional[int] = None) -> None:
    """Reset random state to a new seed."""
    if seed is None:
        import time

        seed = int(time.time() * 1000) % 2**32
    set_seed(seed)
