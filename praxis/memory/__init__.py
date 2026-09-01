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
    MemoryDepthBank,
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
    # The same four regimes as mal_energy_quad, spread ALONG the recurrence
    # instead of stacked at every step. Pass p runs core p % 4 and nothing else,
    # so a step costs ONE memory core regardless of how many the bank holds -
    # against the band smear, where the two cheap arms are always on and a step
    # never costs fewer than two. The bank is ordered cheapest-first, so the
    # depth a regime sits at is the price of reaching it: pass 0's core runs on
    # every forward, and the grid cores are only reached when the pass budget
    # goes that deep (a sampled loop count in training, a KL exit at inference).
    # No bandit and no blend - the cores read different depths, so an
    # inverse-surprise share between them would measure depth rather than
    # forecast quality, and routing stays a pure function of current_depth.
    # abstractinator-h's memory; everything else tracks mal_energy_quad.
    "mal_energy_bank": dict(
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
    # ONE memory, gated, at the FIRST recurrent pass only - the Titans-faithful
    # arrangement, after the depth bank measured as a bank of static MLPs.
    #
    # passes=[0]. The depth bank spread four cores along the recurrence and the
    # late ones starved: `*_memory_core_use` read 0.46 / 0.28 / 0.17 / 0.09,
    # which is not a routing decision, it is the halting distribution. Training
    # samples a loop count up front (halting/kl.py:122-133) and eval exits at
    # loop boundaries (:154), so pass 0 is the ONLY station every input reaches,
    # every gradient step trains, and the speculative decoder sees identically
    # on every forward. One memory call per forward instead of ~3.
    #
    # segment_block=4, matching the reference (lucidrains' train_mac.py runs
    # SEQ_LEN 512 on a 4-token grid = 128 chunks). This is the number that
    # decides whether test-time learning happens at all: retrieval reads
    # PRE-write weights, so the writes the model can feel is chunks - 1, and on
    # a 16-token grid this repo's latent lengths (bytes/8) gave 0.70 writes per
    # forward with 62% of forwards getting ZERO. A 4-token grid gives ~5.
    #
    # swish, NOT serpent. Every parameter of the memory net is a fast weight, so
    # a periodic activation puts its per-feature FREQUENCIES in the test-time
    # update - and the energy rule's step is sign-like at a fixed magnitude,
    # which is well-conditioned on a linear map and not on a frequency (sin(a*x)
    # is not locally linear in a). A parameter-free activation keeps the whole
    # fast-weight set linear, which is what the Adam rule's scale-invariance
    # argument assumes, and it hands the trunk a genuinely different function
    # class - everything else here is periodic (Servant experts, ArcHoPE's phase
    # warp, the harmonic head). `swish` is torch's SiLU, the paper's activation;
    # the `silu` key is transformers' copy.
    # mag_energy's twin with the update DIFFERENTIABLE (the paper's own mode).
    #
    # Energy mode detaches the whole test-time update, which severs the outer
    # loss from the memory net: retrieval reads PRE-write weights, so only chunk
    # 0 reads W0 itself and every later chunk reads a detached constant.
    # Measured, the gradient reaching W0 decays as 1/nc - 1.00x / 0.52x / 0.31x
    # / 0.18x at 2 / 4 / 8 / 16 chunks - while standard mode holds it flat
    # (1.00x / 1.00x / 1.01x / 1.03x). So under energy mode W0 is trained to be
    # a good COLD READOUT and never to be a good INITIALIZATION for the update,
    # which is exactly the shape every run in this line reported: gate high,
    # gain high, adapt ~0. We were training a static function and then measuring
    # that it behaved like one.
    #
    # chunk_size 4, NOT segment_block. Standard mode hard-gates segmentation off
    # (`segment and use_energy`), so the update grid is chunk_size - leaving it
    # at 64 would give ONE chunk at this model's latent lengths and a memory
    # that cannot adapt at all. This keeps mag_energy's 4-latent grid by the
    # only route standard mode has.
    #
    # The learned gates come back with it: to_lr / to_momentum / to_decay are
    # theta_t / eta_t / alpha_t from the paper's Eqs. 13-14, all data-dependent,
    # and they are what makes the step size LEARNED rather than a constant the
    # energy rule has to hardcode. The forgetting gate is init-biased to retain
    # (_DECAY_GATE_BIAS); at its old default it erased the memory within a few
    # chunks.
    "mag_standard": dict(
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
    # mag_energy + STITCHED WRITES across linked batch rows.
    #
    # The write span, not the credit path, is the last standing explanation for
    # why this line's memories never learn. A pass writes over its whole
    # sequence, which at patch_size 8 and block_size 64 is 8-64 latents - 64-512
    # bytes. That is enough for grammar and not for a fact, and the trunk cannot
    # afford longer sequences at this model size.
    #
    # It does not have to. The packer already splits long documents across
    # consecutive rows and drains the remainder into the next one; it simply
    # discarded the linkage at the row boundary, because `block_ids` restart at
    # 1 per row and cannot express it. With `row_continues` published, the
    # memory threads its state along a run of linked rows - so the write span
    # becomes the run's total length while the TRUNK still only ever sees one
    # row. Memory horizon is decoupled from trunk sequence length, which no
    # config change can buy.
    #
    # Total work is unchanged (a run of G rows is G batched calls over b/G rows
    # each); what it costs is serialization into G sequential memory calls.
    # Read `memory_run_length` x `memory_chunks` for the span actually written.
    "mag_energy_stitch": dict(
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
    # Stitched writes AND a differentiable update - the only pairing in which a
    # longer write span can actually pay.
    #
    # `mag_energy_stitch` was self-defeating and abstractinator-e measured it:
    # in energy mode the state handed from one row of a run to the next is
    # DETACHED, so a row at position >= 1 reads detached weights at every chunk
    # including chunk 0, and only RUN-START rows give the memory net any
    # gradient at all. Measured ||grad W0|| against unstitched: 0.72x at runs of
    # 2, 0.48x at 4, 0.35x at 8. -e ran at run_length 1.94, i.e. ~0.70x the
    # gradient of its unstitched twin, and duly gave up that twin's advantage.
    # Standard mode keeps a graph through the carried state, so the same sweep
    # reads 0.99x / 0.97x / 0.93x - the span is nearly free.
    #
    # Everything else tracks mag_standard, including `chunk_size: 4` (standard
    # mode has no segmentation, so the grid comes from chunk_size and leaving it
    # at 64 would give one chunk and a dead memory).
    "mag_standard_stitch": dict(
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
    # mag_energy with the test-time write FROZEN (max_lr=0): same module, same
    # gate, same parameters, same step cost - the surprise is still computed, so
    # the governor sees an identical run - and the only thing removed is whether
    # the write lands. THE control the thread never had: -v, -x and -y all
    # confounded "a gated nonlinear module at this depth" with "test-time
    # memory", and -y's own numbers say the split is lopsided (gate 0.55 and
    # gain 2.56 put the branch at ~76% of the output magnitude, while adapt of
    # 0.004 puts the write at 0.4% of the readout). If this matches its live
    # twin, the adaptation is contributing nothing measurable and the honest
    # comparison for anything bigger is against a dense of the same size.
    "mag_energy_static": dict(
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
    "mag_energy": dict(
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
    # mal_energy_dual has no entry by long-standing omission; the two profiles
    # below are the ones the abstractinator thread actually chooses between, so
    # they say what they are rather than dumping a spec dict into the docs.
    "mal_energy_dual": (
        "Two energy-memory cores of opposed function-class regimes - the "
        "serpent-activation MLP (exponential/harmonic) and the EML tree's "
        "log-minus-exponent - run at EVERY recurrent step and combined by a "
        "floored inverse-surprise bandit rather than a loss-trained router, so "
        "neither can be starved before it matures. Two memory forwards and two "
        "test-time updates per step is the price."
    ),
    "mal_energy_bank": (
        "The four regimes of mal_energy_quad spread ALONG the recurrence "
        "instead of stacked at every step: recurrent pass p runs core p % 4 and "
        "nothing else, so a step costs ONE memory core no matter how many the "
        "bank holds. The bank is ordered cheapest-first, so the pass a regime "
        "sits at is the price of reaching it - pass 0's core runs on every "
        "forward, while the grid cores are only reached when the pass budget "
        "goes that deep (a sampled loop count in training, a KL early exit at "
        "inference). No blend and no bandit: each core reads a different "
        "depth's stream, so a share between them would measure depth rather "
        "than forecast quality, and routing stays a pure function of "
        "current_depth. abstractinator-h's memory."
    ),
    "mag": (
        "Memory-as-Gate (Titans): a memory branch run parallel to attention "
        "and blended with it through a learned gate."
    ),
    "mag_standard": (
        "mag_energy with the test-time update differentiable instead of "
        "detached - the paper's own formulation. The outer loss can then see "
        "the memory THROUGH its writes, so the meta-learned weights are trained "
        "as an initialization for the update rather than only as a cold "
        "readout, and the per-token learning rate, momentum and forgetting "
        "gates are learned rather than fixed. Costs the scan trajectory in "
        "VRAM, which is affordable at one memory call per forward."
    ),
    "mag_energy_stitch": (
        "mag_energy with writes stitched across linked batch rows. The packer "
        "splits long documents across consecutive rows; threading the memory "
        "state along such a run makes the write span the run's total length "
        "while the trunk still sees only one row, decoupling the memory's "
        "horizon from the sequence length the model can afford to train on."
    ),
    "mag_standard_stitch": (
        "mag_standard with writes stitched across linked batch rows - the only "
        "pairing where a longer span can pay, because a differentiable update "
        "keeps gradient flowing back across the whole run. Stitching a DETACHED "
        "update instead starves the memory net: only run-start rows train it."
    ),
    "mag_energy_static": (
        "mag_energy with the test-time write frozen at the meta-learned init "
        "(max_lr=0). A control, not a way to run the model: it isolates how "
        "much of a memory profile's benefit comes from the module being a "
        "gated nonlinearity at that depth versus from the memory actually "
        "learning in context. Same parameters and same step cost as its live "
        "twin, so the two differ in exactly one thing."
    ),
    "mag_energy": (
        "One gated memory at the FIRST recurrent pass only, with the detached "
        "(energy) update, a predictive NextLat write target and a 4-token "
        "update grid. The gate makes the model state whether it wants the "
        "memory as a single readable number instead of leaving it to cancel a "
        "full-weight residual add; pass 0 is the only recurrent step every "
        "input reaches and every gradient step trains, so nothing starves the "
        "way a depth-spread bank does; and the fine grid is what gives the "
        "test-time update enough chunks to be visible at all, since retrieval "
        "reads pre-write weights. The memory net is a plain swish MLP - the one "
        "non-periodic function class in an otherwise harmonic model, and the "
        "only kind whose whole fast-weight set is linear maps."
    ),
}

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
