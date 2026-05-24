"""Mono-Forward trainer.

``MonoForwardTrainer`` runs each ``LocalLayer`` in its own distributed
worker, with per-layer projection matrices trained independently. Two
backends are supported:

- **ray** (the default): each layer runs in its own Ray actor, which
  enables multi-host / multi-raylet fan-out but pays ~300-500 MB of
  CUDA context per actor. The multi-GPU / multi-node story lives here.
- **inprocess**: every layer runs inside the driver process, sharing
  a single CUDA context. Single-host only, but eliminates the
  per-actor context tax so very deep models fit on a single GPU.

Both implement the same public surface (``fit``, ``generate``,
checkpoint save/load); pick one via ``--mono-forward-backend``. See
``PLAN.md`` for the design rationale.

Ray itself is an optional extra (``pyproject.toml``
``[project.optional-dependencies].ray``) and is only imported when the
Ray backend is actually used, so ``import praxis`` still works on
Python builds where Ray has no wheels.
"""

from praxis.trainers.mono_forward.inprocess_trainer import InProcessMonoForwardTrainer
from praxis.trainers.mono_forward.inprocess_worker import LocalLayerWorker
from praxis.trainers.mono_forward.projection import ProjectionMatrix
from praxis.trainers.mono_forward.trainer import MonoForwardTrainer

# ``LayerActor`` is deliberately NOT re-exported here: importing it
# triggers ``import ray`` at module load time, and the point of the
# in-process backend is to let pure-PyTorch runs avoid that. Callers
# that need it can import from :mod:`praxis.trainers.mono_forward.actor`
# directly.

__all__ = [
    "InProcessMonoForwardTrainer",
    "LocalLayerWorker",
    "MonoForwardTrainer",
    "ProjectionMatrix",
]
