"""Callbacks for Praxis training frameworks.

This module provides callbacks for various training frameworks.
Currently supports PyTorch Lightning, with potential for future framework support.
"""

# Generic callbacks (framework-agnostic)
from praxis.callbacks.builder import build_training_callbacks
from praxis.callbacks.printing_progress import (
    PrintingProgressBar,
    create_printing_progress_bar,
)

# Framework-specific imports
try:
    from praxis.callbacks.lightning import (
        AccumulationSchedule,
        BrierLMCallback,
        DynamicsLoggerCallback,
        MetricsLoggerCallback,
        PeriodicEvaluation,
        TerminalInterface,
    )
    from praxis.callbacks.lightning.signal_handler import SignalHandlerCallback

    _HAS_LIGHTNING = True
except ImportError:
    _HAS_LIGHTNING = False


# Base exports (always available)
__all__ = [
    "PrintingProgressBar",
    "create_printing_progress_bar",
    "build_training_callbacks",
]

# Re-export Lightning callbacks for backward compatibility
if _HAS_LIGHTNING:
    __all__.extend(
        [
            "AccumulationSchedule",
            "BrierLMCallback",
            "DynamicsLoggerCallback",
            "MetricsLoggerCallback",
            "PeriodicEvaluation",
            "TerminalInterface",
            "SignalHandlerCallback",
        ]
    )
