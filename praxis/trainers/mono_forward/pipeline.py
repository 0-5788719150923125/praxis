"""Pipeline driver stub.

Phase 3 lands two variants here: a manual ``ray.wait``-based driver loop and
a ``ray.dag`` compiled variant. Phase 1 just reserves the module path.
"""

from __future__ import annotations

_PHASE3_MSG = (
    "Pipeline driver lands in Phase 3 of the Ray Mono-Forward project. "
    "See PROJECT_PLAN.md."
)


def run_manual_pipeline(*args, **kwargs):  # pragma: no cover - Phase 3
    raise NotImplementedError(_PHASE3_MSG)


def run_compiled_pipeline(*args, **kwargs):  # pragma: no cover - Phase 3
    raise NotImplementedError(_PHASE3_MSG)
