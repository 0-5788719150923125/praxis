"""Trainer modules for Praxis."""

from praxis.trainers.backpropagation import BackpropagationTrainer
from praxis.trainers.base import (
    BaseCallback,
    BaseLogger,
    BaseTrainer,
    TrainerConfig,
    reset_random_state,
    set_seed,
)
from praxis.trainers.compile import try_compile
from praxis.trainers.datamodule import BaseDataModule
from praxis.trainers.factory import (
    create_checkpoint_callback,
    create_logger,
    create_progress_callback,
    create_trainer_with_module,
    disable_warnings,
)
from praxis.trainers.module import BaseTrainingModule
from praxis.trainers.progress import BaseProgressBar, get_progress_bar_base
from praxis.trainers.runtime import (
    assemble_trainer,
    print_training_banner,
    resolve_training_logger,
    run_training,
)
from praxis.trainers.ray_support import ensure_ray
from praxis.trainers.seed import reset_seed, seed_everything
from praxis.trainers.setup import (
    ModelBundle,
    assemble_model,
    build_model_info,
    configure_torch_precision,
    register_praxis_models,
    setup_environment,
)
from praxis.trainers.trainer import Trainer

# Framework-specific imports with graceful fallback
try:
    from praxis.trainers.lightning_trainer import (
        LightningModule,
        LightningTrainerWrapper,
        create_csv_logger,
        create_lightning_trainer,
        create_model_checkpoint,
        create_progress_bar,
    )
    from praxis.trainers.lightning_trainer import (
        disable_warnings as disable_warnings_lightning,
    )
    from praxis.trainers.lightning_trainer import (
        reset_seed_lightning,
        seed_everything_lightning,
    )

    _HAS_LIGHTNING = True
except ImportError:
    _HAS_LIGHTNING = False
    LightningModule = None
    LightningTrainerWrapper = None


# Registry for trainers. Each ``mono_forward*`` entry is a profile
# that picks a worker backend in addition to the trainer math:
#
# - ``mono_forward``: in-process, single CUDA context, single host.
#   The default Mono-Forward profile - no Ray dependency, suitable
#   for iterating on very deep models on one GPU.
# - ``mono_forward_ray``: one Ray actor per layer. Multi-host /
#   multi-raylet capable, but pays ~300-500 MB of CUDA context per
#   actor; use this when you actually need multiple machines.
#
# Both are lazy-loaded so selecting ``backpropagation`` doesn't
# import the Mono-Forward package or its (Ray-backed) worker
# runtime; ``mono_forward_ray`` additionally defers ``import ray``
# until ``fit()`` so the in-process profile loads on Python builds
# where Ray has no wheels.
def _get_mono_forward_inprocess_trainer():
    """Lazy load the in-process Mono-Forward trainer."""
    from praxis.trainers.mono_forward import InProcessMonoForwardTrainer

    return InProcessMonoForwardTrainer


def _get_mono_forward_ray_trainer():
    """Lazy load the Ray-backed Mono-Forward trainer."""
    from praxis.trainers.mono_forward import MonoForwardTrainer

    return MonoForwardTrainer


TRAINER_REGISTRY = {
    "backpropagation": BackpropagationTrainer,
    "mono_forward": _get_mono_forward_inprocess_trainer,
    "mono_forward_ray": _get_mono_forward_ray_trainer,
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
    "try_compile",
    # Run setup / lifecycle
    "ensure_ray",
    "setup_environment",
    "register_praxis_models",
    "configure_torch_precision",
    "assemble_model",
    "build_model_info",
    "ModelBundle",
    "assemble_trainer",
    "resolve_training_logger",
    "print_training_banner",
    "run_training",
    # Trainers
    "BackpropagationTrainer",
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
