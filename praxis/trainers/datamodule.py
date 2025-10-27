"""Generic data module interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from torch.utils.data import DataLoader


class BaseDataModule(ABC):
    """Abstract base class for data modules.

    This provides a framework-agnostic interface for data loading.
    Implementations can wrap this for specific frameworks like PyTorch Lightning.
    """

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for the given stage.

        Args:
            stage: One of 'fit', 'validate', 'test', 'predict'
        """
        pass

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        pass

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        pass

    def test_dataloader(self) -> Optional[DataLoader]:
        """Return the test dataloader."""
        return None

    def predict_dataloader(self) -> Optional[DataLoader]:
        """Return the prediction dataloader."""
        return None

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after the given stage.

        Args:
            stage: One of 'fit', 'validate', 'test', 'predict'
        """
        pass
