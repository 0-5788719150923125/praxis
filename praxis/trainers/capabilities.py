"""Trainer capabilities system for defining trainer-specific features and limitations."""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class TrainerCapabilities:
    """
    Defines the capabilities and limitations of a trainer type.

    This allows the system to adapt its behavior based on what each trainer supports,
    rather than hardcoding checks for specific trainer types throughout the codebase.
    """

    # Optimization capabilities
    supports_automatic_optimization: bool = True
    supports_gradient_clipping: bool = True
    supports_gradient_accumulation: bool = True

    # Training capabilities
    supports_mixed_precision: bool = True
    supports_distributed_training: bool = True

    # Callback capabilities
    supports_accumulation_schedule: bool = True

    # Special initialization requirements
    requires_custom_init: bool = False
    requires_optimizer_config: bool = False

    # Lightning-specific
    is_lightning_module: bool = False  # If the trainer itself is a LightningModule
    uses_manual_optimization: bool = False


# Registry of trainer capabilities
TRAINER_CAPABILITIES: Dict[str, TrainerCapabilities] = {
    "backpropagation": TrainerCapabilities(
        is_lightning_module=True,
    ),
    "mono_forward": TrainerCapabilities(
        supports_automatic_optimization=False,
        supports_gradient_clipping=False,
        supports_accumulation_schedule=False,
        requires_custom_init=True,
        requires_optimizer_config=True,
        uses_manual_optimization=True,
    ),
    "pipeline": TrainerCapabilities(
        supports_automatic_optimization=False,
        supports_gradient_clipping=False,
        supports_accumulation_schedule=False,
        requires_custom_init=True,
        uses_manual_optimization=True,
    ),
    # Default capabilities for unknown trainer types
    "default": TrainerCapabilities(),
}


def get_trainer_capabilities(trainer_type: str) -> TrainerCapabilities:
    """
    Get the capabilities for a specific trainer type.

    Args:
        trainer_type: The type of trainer

    Returns:
        TrainerCapabilities object defining what the trainer supports
    """
    return TRAINER_CAPABILITIES.get(trainer_type, TRAINER_CAPABILITIES["default"])


def trainer_supports(trainer_type: str, capability: str) -> bool:
    """
    Check if a trainer supports a specific capability.

    Args:
        trainer_type: The type of trainer
        capability: The capability to check (as attribute name)

    Returns:
        True if the trainer supports the capability, False otherwise
    """
    capabilities = get_trainer_capabilities(trainer_type)
    return getattr(capabilities, capability, False)
