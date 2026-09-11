"""Titans-style learned long-term memory (Behrouz et al. 2024).

A single ``--memory-type`` flag selects a named profile from
the ``memory`` registry; each profile bundles a surfacing strategy and the memory
hyperparameters, so new variants never add CLI arguments. ``"none"`` (default)
disables the module entirely.
"""

from typing import Dict, Optional, Type

from torch import nn

from praxis import registry
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
    MemoryDepthBank,
    MemoryDualSmear,
)
from praxis.registry import Entry

registry.declare(
    "memory",
    title="Long-term memory",
    doc=(
        (
            "Titans-style test-time-learned memory modules (Behrouz et al. 2024), surfaced "
            "as a layer (MAL) or a gate (MAG). A memory is built inside every transformer "
            "block and runs between the attention and feedforward sublayers, carrying its "
            "state across depth steps. Each profile bundles a surfacing and its "
            "hyperparameters, so new variants never add flags; the ``surfacing`` key picks "
            "the implementing module. Unrelated to ``--memory``, which puts a compressive "
            "memory inside attention."
        )
    ),
    entries={
        "none": Entry(
            None,
            "Disabled. The model carries no long-term memory module.",
        ),
        "mal": Entry(
            dict(
                surfacing="mal",
                dense="mlp",
                layers=2,
                expansion=1.0,
                chunk_size=64,
                momentum=True,
                activation="mish",
                parallel_scan=False,
            ),
            (
                "Memory-as-Layer (Titans): a test-time-learned memory MLP applied as "
                "its own residual sub-layer within each transformer block."
            ),
        ),
        "mal_energy": Entry(
            dict(
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
            (
                "Memory-as-Layer with a detached (energy-based) test-time update: the "
                "memory learns by a local surprise rule with no second-order graph, "
                "for much lower VRAM. The update is a fixed Adam-style rule rather "
                "than learned gates, and the key projection is tied to the query "
                "projection, so addressing trains on the task. Each key stores the "
                "next latent (a predictive NextLat target) rather than reconstructing "
                "the current token, so retrieval carries information the residual "
                "stream does not already hold. The update grid is segmented at "
                "surprise spikes, so a context shift starts a fresh write."
            ),
        ),
        "mal_energy_serpent": Entry(
            dict(
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
            (
                "mal_energy with a Serpent memory activation. Its learnable "
                "per-feature frequencies join the test-time fast weights, so the "
                "surprise update tunes the memory's harmonic geometry online as well "
                "as its linear maps, and the memory works in the same oscillatory "
                "basis as the harmonic codec latents it stores."
            ),
        ),
        "mal_energy_dual": Entry(
            dict(
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
            (
                "Two energy-memory cores of opposed function classes - the Serpent MLP "
                "and the EML tree's log-minus-exponent regime - run at every recurrent "
                "step and blended by a floored bandit over each core's forecast "
                "quality (surprise). The bandit is detached from the LM gradient, so "
                "neither core is starved before it matures. Read memory_blend_b: a "
                "rise above 0.5 means the EML core earns its place. Costs two memory "
                "forwards and two test-time updates per step."
            ),
        ),
        "mal_energy_triple": Entry(
            dict(
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
            (
                "mal_energy_dual plus a third core: a KAN whose radial basis centers "
                "are log-magnitude spaced, a coarse-to-fine cascade over the amplitude "
                "axis. The KAN replicates its spline matrix per chunk as a fast "
                "weight, so it runs on every fourth recurrent step only; on skipped "
                "steps the blend renormalizes over the two cheap cores. The bandit "
                "floors every core."
            ),
        ),
        "mal_energy_quad": Entry(
            dict(
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
            (
                "mal_energy_triple plus a fourth core: a spline whose knot positions "
                "and widths are fast weights, so the test-time update moves resolution "
                "to where the sequence is complex. It is the learned-placement "
                "counterpart to the KAN's fixed geometric grid, with the same basis "
                "count and bandit, so the blend weights compare fixed and learned "
                "placement directly. The two grid cores run on staggered phases of a "
                "period-4 cycle, at most one per step."
            ),
        ),
        "mal_energy_bank": Entry(
            dict(
                surfacing="depth_bank",
                dense="mlp",
                dense_b="eml_tree",
                dense_c="kan",
                dense_d="spline",
                num_grids=6,
                grid_spacing="geometric",
                num_knots=6,
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
            (
                "The four cores of mal_energy_quad spread along the recurrence instead "
                "of stacked at every step: pass p runs core p % 4 and nothing else, so "
                "a step costs one memory core however many the bank holds. The bank is "
                "ordered cheapest first, so the grid cores run only when the pass "
                "budget goes that deep. There is no blend and no bandit: the cores "
                "read different depths, so a share between them would measure depth "
                "rather than forecast quality."
            ),
        ),
        "mag": Entry(
            dict(
                surfacing="mag",
                dense="mlp",
                layers=2,
                expansion=1.0,
                chunk_size=64,
                momentum=True,
                activation="mish",
                parallel_scan=False,
            ),
            (
                "Memory-as-Gate (Titans): a memory branch run parallel to attention "
                "and blended with it through a learned gate."
            ),
        ),
        "mag_standard": Entry(
            dict(
                surfacing="mag",
                passes=[0],
                dense="mlp",
                layers=2,
                expansion=0.5,
                chunk_size=4,
                momentum=True,
                activation="swish",
                use_energy=False,
                parallel_scan=True,
                write_objective="predictive",
            ),
            (
                "mag_energy with a differentiable test-time update, the paper's own "
                "formulation. The outer loss sees the memory through its writes, so "
                "the meta-learned weights train as an initialization for the update "
                "rather than only as a cold readout, and the per-token learning rate, "
                "momentum and forgetting gates are learned. chunk_size is 4 because "
                "this mode does not segment, so the update grid is the chunk grid. "
                "Costs the scan trajectory in VRAM, affordable at one memory call per "
                "forward."
            ),
        ),
        "mag_energy_stitch": Entry(
            dict(
                surfacing="mag",
                passes=[0],
                stitch=True,
                dense="mlp",
                layers=2,
                expansion=0.5,
                chunk_size=64,
                segment_block=4,
                momentum=True,
                activation="swish",
                use_energy=True,
                segment=True,
                parallel_scan=True,
                write_objective="predictive",
            ),
            (
                "mag_energy with writes stitched across linked batch rows. The packer "
                "splits long documents across consecutive rows; threading the memory "
                "state along such a run makes the write span the run's total length "
                "while the trunk still sees one row, which decouples the memory's "
                "horizon from the sequence length the trunk can afford. Total work is "
                "unchanged; the cost is running a run's rows as sequential memory "
                "calls. Read memory_run_length x memory_chunks for the span written."
            ),
        ),
        "mag_energy_stitch_adaptive": Entry(
            dict(
                surfacing="mag",
                passes=[0],
                stitch=True,
                dense="mlp",
                layers=2,
                expansion=0.5,
                chunk_size=64,
                segment_block=4,
                momentum=True,
                activation="swish",
                use_energy=True,
                segment=True,
                parallel_scan=True,
                write_objective="predictive",
                write_gate="adaptive",
            ),
            (
                "mag_energy_stitch with the write gated to a target share. A token "
                "writes when its surprise ranks in the top target fraction of the real "
                "tokens at or before it - a causal rank, which depends on their order "
                "alone - and a slow-over-fast surprise EMA moves the target, bounded "
                "in [0, 1], down as the memory's forecasts improve. A rank holds the "
                "share near its target however the surprise distribution drifts, where "
                "a bar keyed to the score's level swings between writing everything "
                "and writing nothing. Read memory_write_share against "
                "memory_write_target; memory_write_selectivity says whether the gate "
                "picks on content."
            ),
        ),
        "mag_energy_stitch_gated": Entry(
            dict(
                surfacing="mag",
                passes=[0],
                stitch=True,
                dense="mlp",
                layers=2,
                expansion=0.5,
                chunk_size=64,
                segment_block=4,
                momentum=True,
                activation="swish",
                use_energy=True,
                segment=True,
                parallel_scan=True,
                write_objective="predictive",
                write_gate="threshold",
            ),
            (
                "mag_energy_stitch with a per-token write gate. Energy mode has no "
                "learned write gate, so every token writes the same fixed-magnitude "
                "step and the memory converges on a mean over the stream. Here a token "
                "writes only when its surprise exceeds a ratio of the sequence's "
                "causal running mean, tilted by a slow-over-fast surprise EMA, so a "
                "chunk where nothing clears the bar holds its weights. It trades write "
                "volume for selectivity and saves no compute. Read memory_write_share "
                "beside memory_write_selectivity."
            ),
        ),
        "mag_standard_stitch": Entry(
            dict(
                surfacing="mag",
                passes=[0],
                stitch=True,
                dense="mlp",
                layers=2,
                expansion=0.5,
                chunk_size=4,
                momentum=True,
                activation="swish",
                use_energy=False,
                parallel_scan=True,
                write_objective="predictive",
            ),
            (
                "mag_standard with writes stitched across linked batch rows, the "
                "pairing in which a longer write span can pay. A detached update hands "
                "a detached state between rows, so only run-start rows would train the "
                "memory net; a differentiable update keeps the graph through the "
                "carried state. Everything else tracks mag_standard, including "
                "chunk_size 4."
            ),
        ),
        "mag_energy_static": Entry(
            dict(
                surfacing="mag",
                passes=[0],
                max_lr=0.0,
                dense="mlp",
                layers=2,
                expansion=0.5,
                chunk_size=64,
                segment_block=4,
                momentum=True,
                activation="swish",
                use_energy=True,
                segment=True,
                parallel_scan=True,
                write_objective="predictive",
            ),
            (
                "mag_energy with the test-time write frozen at the meta-learned init "
                "(max_lr=0). A control: same module, gate, parameters and step cost, "
                "with the surprise still computed, so it differs from its live twin "
                "only in whether the write lands. It separates a gated nonlinear "
                "module at that depth from memory that learns in context; if the two "
                "match, compare anything bigger against a dense layer of the same "
                "size."
            ),
        ),
        "mag_energy": Entry(
            dict(
                surfacing="mag",
                passes=[0],
                dense="mlp",
                layers=2,
                expansion=0.5,
                chunk_size=64,
                segment_block=4,
                momentum=True,
                activation="swish",
                use_energy=True,
                segment=True,
                parallel_scan=True,
                write_objective="predictive",
            ),
            (
                "One gated memory at the first recurrent pass only, with the detached "
                "(energy) update, a predictive NextLat write target and a 4-token "
                "update grid. Pass 0 is the only step every input reaches, so the "
                "memory runs once per forward and nothing starves; the fine grid gives "
                "the update enough chunks to matter, since retrieval reads pre-write "
                "weights. The memory net is a swish MLP, whose fast weights are all "
                "linear maps - the case the energy rule's sign-like step is "
                "well-conditioned for."
            ),
        ),
    },
)

# Internal: surfacing key -> module. Selection flows through the profiles
# above, not this map.
_SURFACINGS: Dict[str, Type[nn.Module]] = {
    "mal": MemoryAsLayer,
    "mag": MemoryAsGate,
    "dual_smear": MemoryBandSmear,  # N=2 (back-compat name)
    "band_smear": MemoryBandSmear,  # N arms
    "depth_bank": MemoryDepthBank,  # N arms, one per recurrent pass
}


def get_memory_profile(name: str) -> Optional[dict]:
    """Resolve a ``--memory-type`` name to its profile spec (None disables)."""
    if name not in registry.namespace("memory"):
        raise ValueError(
            f"Unknown memory profile '{name}'. Choices: {sorted(registry.namespace("memory"))}"
        )
    spec = registry.lookup("memory", name)
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
    "get_memory_profile",
    "build_memory",
    "MemoryBase",
    "NeuralMemory",
    "NeuralMemState",
    "build_memory_model",
    "mem_state_detach",
]
