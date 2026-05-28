"""Factory functions for creating trainer components."""

import logging
import os
from typing import Any, Dict, List, Optional, Union

from .capabilities import get_trainer_capabilities


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


def create_trainer_with_module(
    trainer_type: str,
    model: Any,
    optimizer: Any = None,
    scheduler: Any = None,
    hparams: Dict[str, Any] = None,
    tokenizer: Any = None,
    cache_dir: str = None,
    ckpt_path: str = None,
    trainer_params: Dict[str, Any] = None,
    **kwargs,
) -> tuple[Any, Any]:
    """Create a trainer with appropriate module wrapping.

    This factory function handles all the complexity of different trainer types,
    including LightningModule wrapping for BackpropagationTrainer.

    Args:
        trainer_type: Type of trainer to create
        model: The model to train
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        hparams: Hyperparameters dict
        tokenizer: Tokenizer instance
        cache_dir: Cache directory for checkpoints
        ckpt_path: Checkpoint path to resume from
        trainer_params: Parameters for the underlying trainer (e.g., Lightning Trainer)
        **kwargs: Additional arguments

    Returns:
        Tuple of (trainer, training_module) where training_module is what gets passed to fit()
    """
    from praxis.trainers import TRAINER_REGISTRY
    from praxis.trainers.trainer import Trainer

    if trainer_type not in TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown trainer type '{trainer_type}'. Available trainers: {list(TRAINER_REGISTRY.keys())}"
        )

    trainer_class = TRAINER_REGISTRY[trainer_type]
    # Handle lazy loading functions
    if callable(trainer_class) and not isinstance(trainer_class, type):
        trainer_class = trainer_class()  # Call the lazy loader
    print(f"[TRAINER] Using {trainer_type} trainer: {trainer_class.__name__}")

    # Handle different trainer types based on capabilities
    capabilities = get_trainer_capabilities(trainer_type)

    if trainer_type == "backpropagation":
        # BackpropagationTrainer is a LightningModule, needs special handling
        from praxis.trainers.backpropagation import BackpropagationTrainer

        # Tokenizer-driven gate for byte-token-specific behavior; callers
        # may pass it explicitly, otherwise default to False (the trainer
        # itself does not derive from the encoder anymore).
        byte_level = bool(kwargs.get("byte_level", False))

        # Create the LightningModule wrapper
        lightning_module = BackpropagationTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            hparams=hparams or {},
            tokenizer=tokenizer,
            byte_level=byte_level,
        )

        # Create the actual Lightning Trainer
        trainer = Trainer(**(trainer_params or {}))

        # Return trainer and the module to be passed to fit()
        return trainer, lightning_module

    elif trainer_type in ("mono_forward", "mono_forward_ray"):
        # Both Mono-Forward profiles share the same factory contract:
        # they ARE the trainer (not a LightningModule wrapped by a
        # Trainer), they read what they need from ``trainer_params``,
        # and they silently ignore Lightning-specific keys
        # (accelerator, devices, precision, etc.). The only thing that
        # changes between the profiles is which trainer class
        # ``TRAINER_REGISTRY`` resolved to above.
        merged: Dict[str, Any] = dict(trainer_params or {})
        merged["cache_dir"] = cache_dir
        if tokenizer is not None:
            merged["tokenizer"] = tokenizer
        for key, value in kwargs.items():
            if value is not None:
                merged[key] = value

        trainer = trainer_class(**merged)
        # Return the raw PraxisForCausalLM as the training module - the
        # MonoForwardTrainer's ``fit`` method reaches into
        # ``model.decoder.locals``, ``model.embeds``, and ``model.head``
        # directly, so it wants the unwrapped model, not a Lightning
        # wrapper around it.
        return trainer, model

    else:
        # Standard trainer initialization
        trainer = trainer_class(**(trainer_params or {}))
        return trainer, model
