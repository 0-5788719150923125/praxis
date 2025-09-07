"""Factory functions for creating trainer components."""

from typing import Any, Dict, List, Optional, Union
import logging
import os


def create_logger(
    log_dir: str = ".",
    name: str = "default",
    version: Optional[Union[int, str]] = None,
    format: str = "csv",
    **kwargs,
) -> Any:
    """Create a logger for training.

    This is a generic interface that delegates to framework-specific implementations.

    Args:
        log_dir: Directory to save logs
        name: Name for the logger
        version: Version for the logs
        format: Logger format (csv, tensorboard, etc.)
        **kwargs: Additional logger-specific arguments

    Returns:
        Logger instance
    """
    try:
        from praxis.trainers.lightning_trainer import create_csv_logger

        if format == "csv":
            return create_csv_logger(
                save_dir=log_dir, name=name, version=version, **kwargs
            )
    except ImportError:
        pass

    # Fallback to basic Python logger
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def create_checkpoint_callback(
    dirpath: Optional[str] = None,
    filename: Optional[str] = None,
    monitor: Optional[str] = None,
    mode: str = "min",
    save_last: Optional[bool] = None,
    save_top_k: int = 1,
    **kwargs,
) -> Any:
    """Create a checkpoint callback.

    This is a generic interface that delegates to framework-specific implementations.

    Args:
        dirpath: Directory to save checkpoints
        filename: Checkpoint filename format
        monitor: Metric to monitor
        mode: Minimization or maximization mode
        save_last: Whether to save last checkpoint
        save_top_k: Number of top checkpoints to save
        **kwargs: Additional callback-specific arguments

    Returns:
        Checkpoint callback instance
    """
    try:
        from praxis.trainers.lightning_trainer import create_model_checkpoint

        return create_model_checkpoint(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_last=save_last,
            save_top_k=save_top_k,
            **kwargs,
        )
    except ImportError:
        # Return a dummy callback that does nothing
        class DummyCheckpoint:
            def __init__(self, **kwargs):
                pass

        return DummyCheckpoint(**kwargs)


def create_progress_callback(
    refresh_rate: int = 1, leave: bool = False, **kwargs
) -> Any:
    """Create a progress bar callback.

    This is a generic interface that delegates to framework-specific implementations.

    Args:
        refresh_rate: How often to refresh
        leave: Whether to leave progress bar after completion
        **kwargs: Additional callback-specific arguments

    Returns:
        Progress callback instance
    """
    try:
        from praxis.trainers.lightning_trainer import create_progress_bar

        return create_progress_bar(refresh_rate=refresh_rate, leave=leave, **kwargs)
    except ImportError:
        # Return a dummy callback
        class DummyProgress:
            def __init__(self, **kwargs):
                pass

        return DummyProgress(**kwargs)


def disable_warnings() -> None:
    """Disable framework-specific warnings."""
    # Try Lightning first
    try:
        from praxis.trainers.lightning_trainer import (
            disable_warnings as disable_lightning,
        )

        disable_lightning()
    except ImportError:
        pass

    # Disable common warnings
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    # Disable logging warnings
    logging.getLogger("pytorch").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
