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
    MemoryBandSmear,
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
    # mal_energy_dual + a THIRD memory core: a geometric-grid KAN (dense_c=kan).
    # Its RBF centers are log-magnitude spaced with per-center widths - a
    # coarse-to-fine radial cascade ("fractal zoom") over the amplitude axis,
    # rather than the codec's harmonic basis (A) or the EML log-minus-exponent
    # regime (B). num_grids is kept small (6) because a KAN memory net replicates
    # its spline matrix per chunk as a fast weight; a geometric grid resolves the
    # dynamic range with fewer centers, keeping that cost near the other cores.
    # The bandit floors every arm, so the KAN can't be starved before it matures.
    # Third module of abstractinator-c; everything else tracks mal_energy_dual.
    "mal_energy_triple": dict(
        surfacing="band_smear",
        dense="mlp",
        dense_b="eml_tree",
        dense_c="kan",
        num_grids=6,
        grid_spacing="geometric",
        # Sparse KAN: the costly third core fires only at the 4th recurrent step
        # and every 4th after (current_depth % 4 == 3) - 5 of 21 depths - so it
        # runs ~1/4 as often. The two cheap cores (energy, EML) stay dense; on
        # skipped steps the blend renormalizes over just those two.
        kan_sparse=dict(period=4, phase=3),
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
    # mal_energy_triple + a FOURTH memory core: a learned-knot spline
    # (dense_d=spline). Compact-support hat basis whose knot positions AND
    # widths are nn.Parameters - as memory fast weights, the test-time surprise
    # update MOVES THE KNOTS, so resolution concentrates where the sequence is
    # complex and coarsens where it is smooth. The adaptive-resolution
    # counterpart to arm C's deliberately-frozen geometric grid: same basis
    # count (6), same bandit, so the blend weights measure fixed vs learned
    # placement head-to-head. The two grid-replicating cores fire on staggered
    # phases of the same period-4 cycle (at most one expensive core per
    # recurrent step), keeping step cost near the triple's. Fourth module of
    # abstractinator-d; everything else tracks mal_energy_triple.
    "mal_energy_quad": dict(
        surfacing="band_smear",
        dense="mlp",
        dense_b="eml_tree",
        dense_c="kan",
        dense_d="spline",
        num_grids=6,
        grid_spacing="geometric",
        num_knots=6,
        sparse=dict(
            kan=dict(period=4, phase=3),
            spline=dict(period=4, phase=1),
        ),
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
    "mal_energy_triple": (
        "mal_energy_dual plus a third memory core: a geometric-grid KAN whose "
        "radial basis centers are log-magnitude spaced with per-center widths - a "
        "coarse-to-fine ('fractal zoom') cascade over the amplitude axis. Three "
        "opposed function-class regimes (harmonic energy, EML log-minus-exponent, "
        "multi-scale radial) compete under one floored inverse-surprise bandit, "
        "so none can be starved by the LM loss. abstractinator-c's memory."
    ),
    "mal_energy_quad": (
        "mal_energy_triple plus a fourth memory core: a learned-knot spline "
        "whose compact-support hat basis has its knot positions and widths as "
        "fast weights - the test-time surprise update re-knots the basis "
        "online, concentrating resolution where the sequence is complex. The "
        "adaptive-resolution counterpart to the KAN arm's fixed geometric "
        "grid; the floored bandit measures fixed vs learned placement head-to-"
        "head, with the two grid cores firing on staggered sparse phases so "
        "per-step cost stays near the triple. abstractinator-d's memory."
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
    "dual_smear": MemoryBandSmear,  # N=2 (back-compat name)
    "band_smear": MemoryBandSmear,  # N arms
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
