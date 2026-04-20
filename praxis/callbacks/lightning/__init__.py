"""PyTorch Lightning-specific callbacks for Praxis.

These callbacks are designed to work with PyTorch Lightning training framework.
"""

from praxis.callbacks.lightning.accumulation import AccumulationSchedule
from praxis.callbacks.lightning.brier_lm import BrierLMCallback
from praxis.callbacks.lightning.dynamics import DynamicsLoggerCallback
from praxis.callbacks.lightning.evaluation import PeriodicEvaluation
from praxis.callbacks.lightning.metrics import MetricsLoggerCallback
from praxis.callbacks.lightning.terminal import TerminalInterface

# Registry for Lightning callbacks
LIGHTNING_CALLBACK_REGISTRY = {
    "periodic_evaluation": PeriodicEvaluation,
    "terminal_interface": TerminalInterface,
    "accumulation_schedule": AccumulationSchedule,
    "metrics_logger": MetricsLoggerCallback,
    "dynamics_logger": DynamicsLoggerCallback,
    "brier_lm": BrierLMCallback,
}

__all__ = [
    "PeriodicEvaluation",
    "TerminalInterface",
    "AccumulationSchedule",
    "MetricsLoggerCallback",
    "DynamicsLoggerCallback",
    "BrierLMCallback",
    "LIGHTNING_CALLBACK_REGISTRY",
]
