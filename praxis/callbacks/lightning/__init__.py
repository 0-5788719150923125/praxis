"""PyTorch Lightning-specific callbacks for Praxis.

These callbacks are designed to work with PyTorch Lightning training framework.
"""

from praxis.callbacks.lightning.accumulation import AccumulationSchedule
from praxis.callbacks.lightning.brier_lm import BrierLMCallback
from praxis.callbacks.lightning.dynamics import DynamicsLoggerCallback
from praxis.callbacks.lightning.evaluation import PeriodicEvaluation
from praxis.callbacks.lightning.harmonic_weight_rl import HarmonicWeightRLCallback
from praxis.callbacks.lightning.memory_profiler import MemoryProfilerCallback
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
    "memory_profiler": MemoryProfilerCallback,
    "harmonic_weight_rl": HarmonicWeightRLCallback,
}

__all__ = [
    "PeriodicEvaluation",
    "TerminalInterface",
    "AccumulationSchedule",
    "MetricsLoggerCallback",
    "DynamicsLoggerCallback",
    "BrierLMCallback",
    "MemoryProfilerCallback",
    "HarmonicWeightRLCallback",
    "LIGHTNING_CALLBACK_REGISTRY",
]
