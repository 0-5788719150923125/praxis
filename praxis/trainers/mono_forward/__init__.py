"""Mono-Forward trainer.

``MonoForwardTrainer`` runs each ``LocalLayer`` in its own distributed
worker, with the shared output head replicated across workers and
periodically averaged. Each worker trains its layer locally against the
next-token objective, so activation memory is O(1) in depth and gradient
flow never crosses a layer boundary.

The current implementation uses Ray to host and pipeline the workers.
That dependency is an implementation detail: a future port to Lightning,
native ``torch.distributed``, Hivemind, or monarch would slot in behind
the same ``MonoForwardTrainer`` surface without changing the public API
or the training math. The package name and class name deliberately do
NOT carry a ``_ray`` suffix so that downstream code doesn't need to
change when the worker backend does.

Ray itself is an *optional* extra (``pyproject.toml``
``[project.optional-dependencies].ray``) and is imported lazily inside
the trainer / actor constructors, so ``import praxis`` still works on
Python builds where Ray has no wheels.
"""

from praxis.trainers.mono_forward.actor import LayerActor
from praxis.trainers.mono_forward.projection import ProjectionMatrix
from praxis.trainers.mono_forward.trainer import MonoForwardTrainer

__all__ = ["LayerActor", "MonoForwardTrainer", "ProjectionMatrix"]
