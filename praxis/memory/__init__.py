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
from praxis.memory.surfacings import (
    MemoryAsGate,
    MemoryAsLayer,
    MemoryBase,
    MemoryDualSmear,
)

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
        parallel_scan=False,
    ),
    "mal_energy": dict(
        surfacing="mal",
        dense="mlp",
        layers=2,
        expansion=0.5,
        chunk_size=64,
        momentum=True,
        activation="mish",
        use_energy=True,
        segment=True,
        segment_block=16,
        parallel_scan=True,
        write_objective="predictive",
    ),
    # mal_energy with a harmonic (Serpent) memory activation instead of mish.
    # Serpent's per-feature frequencies are learnable, so they join the memory's
    # fast weights: the test-time surprise update tunes the memory's harmonic
    # geometry online, not just its linear maps. Matches the spectral latents the
    # abstractinator harmonic codec produces (mish gave the memory a non-periodic
    # basis mismatched with what it stores). Everything else tracks mal_energy.
    "mal_energy_serpent": dict(
        surfacing="mal",
        dense="mlp",
        layers=2,
        expansion=0.5,
        chunk_size=64,
        momentum=True,
        activation="serpent",
        use_energy=True,
        segment=True,
        segment_block=16,
        parallel_scan=True,
        write_objective="predictive",
    ),
    # Two energy-memory cores of OPPOSED function-class regimes, combined by a
    # REWARD-protected blend (not a loss-trained router, which would starve the
    # granular EML core before it matured). Core A is the serpent memory
    # (exponential energy regime); core B swaps the memory net to the EML tree
    # (dense_b=eml_tree, the log-minus-exponent e^x-Log(y) regime). The blend
    # weight is a self-contained bandit over each core's forecast quality
    # (surprise), detached from the LM gradient and floored so neither regime can
    # collapse - the two are held on a stable axis. Watch memory_blend_b: a slow
    # rise above 0.5 = EML earning its granular keep; a fall to the floor = it is
    # not. Everything else tracks mal_energy_serpent.
    "mal_energy_dual": dict(
        surfacing="dual_smear",
        dense="mlp",
        dense_b="eml_tree",
        layers=2,
        expansion=0.5,
        chunk_size=64,
        momentum=True,
        activation="serpent",
        use_energy=True,
        segment=True,
        segment_block=16,
        parallel_scan=True,
        write_objective="predictive",
    ),
    "mag": dict(
        surfacing="mag",
        dense="mlp",
        layers=2,
        expansion=1.0,
        chunk_size=64,
        momentum=True,
        activation="mish",
        parallel_scan=False,
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
    "mal_energy": (
        "Memory-as-Layer with a detached (energy-based) test-time update: the "
        "memory learns by a local surprise rule with no second-order graph, for "
        "much lower VRAM. The update uses a fixed Adam-style adaptive rule (EMA "
        "1st/2nd moment, constant lr) in place of learned gates; the key "
        "projection is tied to the query projection (so addressing trains on the "
        "task). The write target is predictive (NextLat): each key stores the "
        "*next* latent stream_{t+1} (stop-gradded, Huber surprise) rather than "
        "reconstructing the current token - so retrieval carries belief-state "
        "information the residual stream doesn't already hold, instead of an echo "
        "the model just routes around. The update grid is segmented at surprise "
        "spikes (EM-LLM-style events, capped at chunk_size) so a context shift "
        "starts a fresh memory write."
    ),
    "mal_energy_serpent": (
        "mal_energy with a harmonic Serpent activation in the memory net. Its "
        "learnable per-feature frequencies join the test-time fast weights, so "
        "the surprise update re-tunes the memory's harmonic geometry online - a "
        "second test-time adaptation axis on top of the weight update - and the "
        "memory represents content in the same oscillatory basis as the "
        "abstractinator harmonic codec it stores."
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
    "dual_smear": MemoryDualSmear,
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
