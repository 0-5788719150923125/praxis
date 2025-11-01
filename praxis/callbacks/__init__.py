"""Callbacks for Praxis training frameworks.

This module provides callbacks for various training frameworks.
Currently supports PyTorch Lightning, with potential for future framework support.
"""

import warnings
from typing import Any, Dict, Type

# Generic callbacks (framework-agnostic)
from praxis.callbacks.printing_progress import (
    PrintingProgressBar,
    create_printing_progress_bar,
)

# Framework-specific imports
try:
    from praxis.callbacks.lightning import (
        LIGHTNING_CALLBACK_REGISTRY,
        AccumulationSchedule,
        DynamicsLoggerCallback,
        MetricsLoggerCallback,
        PeriodicEvaluation,
        TerminalInterface,
        TimeBasedCheckpoint,
    )
    from praxis.callbacks.lightning.signal_handler import SignalHandlerCallback

    _HAS_LIGHTNING = True
except ImportError:
    _HAS_LIGHTNING = False
    LIGHTNING_CALLBACK_REGISTRY = {}


def get_callback_registry(framework: str = "lightning") -> Dict[str, Type]:
    """Get the callback registry for a specific framework.

    Args:
        framework: The training framework ("lightning", etc.)

    Returns:
        Dictionary mapping callback names to classes

    Raises:
        ValueError: If framework is not supported
    """
    if framework == "lightning":
        if not _HAS_LIGHTNING:
            raise ImportError(
                "Lightning callbacks not available. "
                "Please ensure PyTorch Lightning is installed."
            )
        return LIGHTNING_CALLBACK_REGISTRY
    else:
        raise ValueError(
            f"Framework '{framework}' not supported. "
            f"Available frameworks: lightning"
        )


# Unified callback registry (defaults to Lightning for backward compatibility)
CALLBACK_REGISTRY = LIGHTNING_CALLBACK_REGISTRY.copy() if _HAS_LIGHTNING else {}

# Base exports (always available)
__all__ = [
    "CALLBACK_REGISTRY",
    "get_callback_registry",
    "PrintingProgressBar",
    "create_printing_progress_bar",
]

# Re-export Lightning callbacks for backward compatibility
if _HAS_LIGHTNING:
    __all__.extend(
        [
            "AccumulationSchedule",
            "DynamicsLoggerCallback",
            "MetricsLoggerCallback",
            "PeriodicEvaluation",
            "TerminalInterface",
            "TimeBasedCheckpoint",
            "SignalHandlerCallback",
        ]
    )
