"""Progress bar utilities for training."""

from typing import Any, Optional


class BaseProgressBar:
    """Base class for progress bars."""
    
    def __init__(self, refresh_rate: int = 1, process_position: int = 0, leave: bool = False):
        """Initialize progress bar.
        
        Args:
            refresh_rate: How often to refresh
            process_position: Position for multi-process
            leave: Whether to leave progress bar after completion
        """
        self.refresh_rate = refresh_rate
        self.process_position = process_position
        self.leave = leave
        self._train_progress_bar = None
        self._val_progress_bar = None
        self._test_progress_bar = None
        self._predict_progress_bar = None
    
    @property
    def train_progress_bar(self):
        """Get training progress bar."""
        return self._train_progress_bar
    
    @property
    def val_progress_bar(self):
        """Get validation progress bar."""
        return self._val_progress_bar
    
    @property
    def test_progress_bar(self):
        """Get test progress bar."""
        return self._test_progress_bar
    
    @property
    def predict_progress_bar(self):
        """Get prediction progress bar."""
        return self._predict_progress_bar


def get_progress_bar_base():
    """Get the appropriate progress bar base class.
    
    Returns the Lightning progress bar if available, otherwise a basic implementation.
    """
    try:
        from lightning.pytorch.callbacks import TQDMProgressBar
        return TQDMProgressBar
    except ImportError:
        return BaseProgressBar