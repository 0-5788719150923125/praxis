"""Titans-style learned long-term memory (Behrouz et al. 2024).

A single ``--memory-type`` flag selects a named profile from
``MEMORY_REGISTRY``; each profile bundles a surfacing strategy and the memory
hyperparameters, so new variants never add CLI arguments. ``"none"`` (default)
disables the module entirely.
"""

from typing import Dict, Optional, Type

from torch import nn

from praxis.memory.models import build_memory_model
from praxis.memory.neural_memory import (
    NeuralMemory,
    NeuralMemState,
    mem_state_detach,
)
from praxis.memory.surfacings import MemoryAsGate, MemoryAsLayer, MemoryBase

# Named profiles. Each value is a spec dict (or None to disable); the
# ``surfacing`` key picks the implementing module from ``_SURFACINGS``.
MEMORY_REGISTRY: Dict[str, Optional[dict]] = {
    "none": None,
    "mal": dict(
        surfacing="mal",
        dense="mlp",
        layers=2,
        expansion=1.0,
        chunk_size=64,
        momentum=True,
        activation="mish",
    ),
    "mag": dict(
        surfacing="mag",
        dense="mlp",
        layers=2,
        expansion=1.0,
        chunk_size=64,
        momentum=True,
        activation="mish",
    ),
}

# Rendered by the auto-docs generator in place of class docstrings, since
# registry values are profile dicts rather than classes.
MEMORY_PROFILE_DESCRIPTIONS: Dict[str, str] = {
    "none": "Disabled. The model carries no long-term memory module.",
    "mal": (
        "Memory-as-Layer (Titans): a test-time-learned memory MLP applied as "
        "its own residual sub-layer within each transformer block."
    ),
    "mag": (
        "Memory-as-Gate (Titans): a memory branch run parallel to attention "
        "and blended with it through a learned gate."
    ),
}

# Internal: surfacing key -> module. Selection flows through the profiles
# above, not this map.
_SURFACINGS: Dict[str, Type[nn.Module]] = {
    "mal": MemoryAsLayer,
    "mag": MemoryAsGate,
}


def get_memory_profile(name: str) -> Optional[dict]:
    """Resolve a ``--memory-type`` name to its profile spec (None disables)."""
    if name not in MEMORY_REGISTRY:
        raise ValueError(
            f"Unknown memory profile '{name}'. Choices: {sorted(MEMORY_REGISTRY)}"
        )
    spec = MEMORY_REGISTRY[name]
    return dict(spec) if spec is not None else None


def build_memory(config) -> nn.Module:
    """Instantiate the memory surfacing for a block, or a no-op when disabled.

    Hyperparameters come from the profile keyed by ``config.memory_type``, not
    from the config itself - the config only carries the profile name.
    """
    spec = get_memory_profile(getattr(config, "memory_type", "none"))
    if not spec:
        return MemoryBase(config)
    return _SURFACINGS[spec["surfacing"]](config, spec)


__all__ = [
    "MEMORY_REGISTRY",
    "MEMORY_PROFILE_DESCRIPTIONS",
    "get_memory_profile",
    "build_memory",
    "MemoryBase",
    "NeuralMemory",
    "NeuralMemState",
    "build_memory_model",
    "mem_state_detach",
]
