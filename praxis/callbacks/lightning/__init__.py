"""PyTorch Lightning-specific callbacks for Praxis.

These callbacks are designed to work with PyTorch Lightning training framework.
"""

from praxis.callbacks.lightning.accumulation import AccumulationSchedule
from praxis.callbacks.lightning.checkpoint import TimeBasedCheckpoint
from praxis.callbacks.lightning.evaluation import PeriodicEvaluation
from praxis.callbacks.lightning.terminal import TerminalInterface

# Registry for Lightning callbacks
LIGHTNING_CALLBACK_REGISTRY = {
    "periodic_evaluation": PeriodicEvaluation,
    "terminal_interface": TerminalInterface,
    "time_based_checkpoint": TimeBasedCheckpoint,
    "accumulation_schedule": AccumulationSchedule,
}

__all__ = [
    "PeriodicEvaluation",
    "TerminalInterface",
    "TimeBasedCheckpoint",
    "AccumulationSchedule",
    "LIGHTNING_CALLBACK_REGISTRY",
]