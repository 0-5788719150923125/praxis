"""PyTorch Lightning-specific callbacks for Praxis.

These callbacks are designed to work with PyTorch Lightning training framework.
"""

from praxis.callbacks.lightning.accumulation import AccumulationSchedule
from praxis.callbacks.lightning.brier_lm import BrierLMCallback
from praxis.callbacks.lightning.dynamics import DynamicsLoggerCallback
from praxis.callbacks.lightning.engagement_live import EngagementLiveRewardCallback
from praxis.callbacks.lightning.evaluation import PeriodicEvaluation
from praxis.callbacks.lightning.harmonic_weight_rl import HarmonicWeightRLCallback
from praxis.callbacks.lightning.host_memory import HostMemoryCallback
from praxis.callbacks.lightning.memory_profiler import MemoryProfilerCallback
from praxis.callbacks.lightning.metrics import MetricsLoggerCallback
from praxis.callbacks.lightning.orchestration import ExpertPoolCallback
from praxis.callbacks.lightning.paper import PaperBuildCallback
from praxis.callbacks.lightning.rlct import RLCTLandscapeCallback
from praxis.callbacks.lightning.spider import SpiderCallback
from praxis.callbacks.lightning.terminal import TerminalInterface

# Registry for Lightning callbacks
LIGHTNING_CALLBACK_REGISTRY = {
    "periodic_evaluation": PeriodicEvaluation,
    "terminal_interface": TerminalInterface,
    "accumulation_schedule": AccumulationSchedule,
    "metrics_logger": MetricsLoggerCallback,
    "dynamics_logger": DynamicsLoggerCallback,
    "rlct_landscape": RLCTLandscapeCallback,
    "brier_lm": BrierLMCallback,
    "memory_profiler": MemoryProfilerCallback,
    "host_memory": HostMemoryCallback,
    "harmonic_weight_rl": HarmonicWeightRLCallback,
    "engagement_live": EngagementLiveRewardCallback,
    "expert_pool": ExpertPoolCallback,
    "paper_build": PaperBuildCallback,
    "spider": SpiderCallback,
}

__all__ = [
    "PeriodicEvaluation",
    "TerminalInterface",
    "AccumulationSchedule",
    "MetricsLoggerCallback",
    "DynamicsLoggerCallback",
    "RLCTLandscapeCallback",
    "BrierLMCallback",
    "MemoryProfilerCallback",
    "HostMemoryCallback",
    "HarmonicWeightRLCallback",
    "EngagementLiveRewardCallback",
    "ExpertPoolCallback",
    "PaperBuildCallback",
    "SpiderCallback",
    "LIGHTNING_CALLBACK_REGISTRY",
]
