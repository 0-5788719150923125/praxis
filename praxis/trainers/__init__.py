"""Trainer modules for Praxis."""

from praxis.trainers.base import (
    BaseTrainer,
    BaseCallback,
    BaseLogger,
    TrainerConfig,
    set_seed,
    reset_random_state,
)
from praxis.trainers.compile import try_compile_model, try_compile_optimizer
from praxis.trainers.backpropagation import BackpropagationTrainer
from praxis.trainers.module import BaseTrainingModule
from praxis.trainers.datamodule import BaseDataModule
from praxis.trainers.seed import seed_everything, reset_seed
from praxis.trainers.trainer import Trainer
from praxis.trainers.factory import (
    create_logger,
    create_checkpoint_callback,
    create_progress_callback,
    disable_warnings,
    create_trainer_with_module,
)
from praxis.trainers.progress import BaseProgressBar, get_progress_bar_base

# Framework-specific imports with graceful fallback
try:
    from praxis.trainers.lightning_trainer import (
        LightningModule,
        LightningTrainerWrapper,
        create_lightning_trainer,
        create_csv_logger,
        create_model_checkpoint,
        create_progress_bar,
        disable_warnings as disable_warnings_lightning,
        seed_everything_lightning,
        reset_seed_lightning,
    )

    _HAS_LIGHTNING = True
except ImportError:
    _HAS_LIGHTNING = False
    LightningModule = None
    LightningTrainerWrapper = None

# Import MonoForward trainer (pipeline version)
from praxis.trainers.mono_forward_pipeline import MonoForwardPipelineModule

# Registry for trainers
TRAINER_REGISTRY = {
    "backpropagation": BackpropagationTrainer,
    "mono_forward": MonoForwardPipelineModule,
}

# Lightning wrapper is not exposed as a separate trainer type


def create_trainer(
    framework: str = "lightning", config: TrainerConfig = None, **kwargs
):
    """Create a trainer for the specified framework.

    Args:
        framework: Training framework to use ("lightning", etc.)
        config: TrainerConfig object with training settings
        **kwargs: Additional framework-specific arguments

    Returns:
        Trainer instance for the specified framework
    """
    if framework == "lightning":
        if not _HAS_LIGHTNING:
            raise ImportError("PyTorch Lightning is required for Lightning trainer")
        return create_lightning_trainer(config or TrainerConfig(), **kwargs)
    else:
        raise ValueError(f"Unknown framework: {framework}")


__all__ = [
    # Base classes
    "BaseTrainer",
    "BaseCallback",
    "BaseLogger",
    "BaseTrainingModule",
    "BaseDataModule",
    "TrainerConfig",
    # Utilities
    "set_seed",
    "reset_random_state",
    "seed_everything",
    "reset_seed",
    "try_compile_model",
    "try_compile_optimizer",
    # Trainers
    "BackpropagationTrainer",
    "MonoForwardPipelineModule",
    "Trainer",
    # Factory functions
    "create_trainer",
    "create_trainer_with_module",
    "create_logger",
    "create_checkpoint_callback",
    "create_progress_callback",
    "disable_warnings",
    "BaseProgressBar",
    "get_progress_bar_base",
    # Registry
    "TRAINER_REGISTRY",
]

# Add Lightning exports if available
if _HAS_LIGHTNING:
    __all__.extend(
        [
            "LightningModule",
            "LightningTrainerWrapper",
            "create_lightning_trainer",
            "create_csv_logger",
            "create_model_checkpoint",
            "create_progress_bar",
            "disable_warnings",
            "seed_everything_lightning",
            "reset_seed_lightning",
        ]
    )
