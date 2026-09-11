"""PyTorch Lightning-specific callbacks for Praxis.

These callbacks are designed to work with PyTorch Lightning training framework.
"""

from praxis import registry
from praxis.callbacks.lightning.accumulation import AccumulationSchedule
from praxis.callbacks.lightning.brier_lm import BrierLMCallback
from praxis.callbacks.lightning.compute_profiler import ComputeProfilerCallback
from praxis.callbacks.lightning.dynamics import DynamicsLoggerCallback
from praxis.callbacks.lightning.engagement_live import EngagementLiveRewardCallback
from praxis.callbacks.lightning.evaluation import PeriodicEvaluation
from praxis.callbacks.lightning.generation_queue import GenerationQueueCallback
from praxis.callbacks.lightning.governor import GNSBatchGovernor
from praxis.callbacks.lightning.harmonic_weight_rl import HarmonicWeightRLCallback
from praxis.callbacks.lightning.host_memory import HostMemoryCallback
from praxis.callbacks.lightning.memory_profiler import MemoryProfilerCallback
from praxis.callbacks.lightning.metrics import MetricsLoggerCallback
from praxis.callbacks.lightning.orchestration import ExpertPoolCallback
from praxis.callbacks.lightning.paper import PaperBuildCallback
from praxis.callbacks.lightning.rlct import RLCTLandscapeCallback
from praxis.callbacks.lightning.snapshot_pump import SnapshotPumpCallback
from praxis.callbacks.lightning.spider import SpiderCallback
from praxis.callbacks.lightning.stall_watchdog import StallWatchdogCallback
from praxis.callbacks.lightning.terminal import TerminalInterface
from praxis.registry import Entry

registry.declare(
    "callbacks",
    entries={
        "periodic_evaluation": PeriodicEvaluation,
        "terminal_interface": TerminalInterface,
        "generation_queue": GenerationQueueCallback,
        "accumulation_schedule": AccumulationSchedule,
        "gns_batch_governor": GNSBatchGovernor,
        "metrics_logger": MetricsLoggerCallback,
        "dynamics_logger": DynamicsLoggerCallback,
        "rlct_landscape": RLCTLandscapeCallback,
        "brier_lm": BrierLMCallback,
        "memory_profiler": MemoryProfilerCallback,
        "compute_profiler": ComputeProfilerCallback,
        "host_memory": HostMemoryCallback,
        "harmonic_weight_rl": Entry(
            HarmonicWeightRLCallback,
            (
                "Drives the harmonic-weight policy from the training loop: every "
                "``period`` steps it samples a harmonic edit to one weight row, lets "
                "training run ``horizon`` steps while integrating an EMA return from "
                "the loss improvement, then rewards the controller and keeps or rolls "
                "back the edit. Order it before ``metrics_logger``."
            ),
        ),
        "engagement_live": Entry(
            EngagementLiveRewardCallback,
            (
                "Drains live web rewards (Print answers, joke approvals) into the "
                "training loop on its own cadence, folding each interaction into the "
                "matching forward-path policy's homeostatic energy, its REINFORCE "
                "baseline. Order it before ``metrics_logger``."
            ),
        ),
        "expert_pool": Entry(
            ExpertPoolCallback,
            (
                "Drives the remote-expert pool during training: starts the Node "
                "sidecar, then each step syncs membership, runs a non-blocking local "
                "update across the pool and routes a vote through it. The pool only "
                "observes - its outputs do not feed the model's activations or loss."
            ),
        ),
        "paper_build": PaperBuildCallback,
        "spider": Entry(
            SpiderCallback,
            (
                "Mirrors the spider's ``spider.db`` counters into the metrics stream "
                "at logging intervals, so the crawl shows up as Research-tab cards. "
                "Read-only: the spider worker stays the sole writer."
            ),
        ),
        "snapshot_pump": SnapshotPumpCallback,
    },
)

__all__ = [
    "PeriodicEvaluation",
    "TerminalInterface",
    "GenerationQueueCallback",
    "AccumulationSchedule",
    "GNSBatchGovernor",
    "MetricsLoggerCallback",
    "DynamicsLoggerCallback",
    "RLCTLandscapeCallback",
    "BrierLMCallback",
    "MemoryProfilerCallback",
    "ComputeProfilerCallback",
    "HostMemoryCallback",
    "StallWatchdogCallback",
    "HarmonicWeightRLCallback",
    "EngagementLiveRewardCallback",
    "ExpertPoolCallback",
    "PaperBuildCallback",
    "SpiderCallback",
    "SnapshotPumpCallback",
]
