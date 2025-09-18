"""Generic training module interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


class BaseTrainingModule(ABC):
    """Abstract base class for training modules.

    This provides a framework-agnostic interface for models that can be trained.
    Implementations can wrap this for specific frameworks like PyTorch Lightning.
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass of the model."""
        pass

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Process a single training batch.

        Returns:
            Dictionary with at least a 'loss' key
        """
        pass

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Process a single validation batch.

        Returns:
            Dictionary with metrics
        """
        pass

    @abstractmethod
    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Process a single test batch.

        Returns:
            Dictionary with metrics
        """
        pass

    @abstractmethod
    def configure_optimizers(self) -> Any:
        """Configure optimizers and schedulers.

        Returns:
            Optimizer(s) and optionally scheduler(s)
        """
        pass

    def on_train_epoch_start(self) -> None:
        """Called at the start of training epoch."""
        pass

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        pass

    def on_validation_epoch_start(self) -> None:
        """Called at the start of validation epoch."""
        pass

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        pass
