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
    # arrangement, and mag_standard is its differentiable-update variant (the
    # paper's own mode) against mag_energy's detached one.
    #
    # passes=[0], because pass 0 is the only station every input reaches:
    # training samples a loop count up front (halting/kl.py:122-133) and eval
    # exits at loop boundaries (:154). One memory call per forward instead of ~3.
    #
    # Energy mode detaches the whole test-time update, which severs the outer
    # loss from the memory net - retrieval reads PRE-write weights, so only chunk
    # 0 reads W0 itself. The gradient reaching W0 decays as 1/nc (1.00x / 0.52x /
    # 0.31x / 0.18x at 2 / 4 / 8 / 16 chunks) where standard mode holds it flat.
    # Under energy mode W0 is trained to be a good COLD READOUT, never a good
    # INITIALIZATION for the update.
    #
    # chunk_size 4, NOT segment_block: standard mode hard-gates segmentation off
    # (`segment and use_energy`), so the update grid is chunk_size, and 64 would
    # give ONE chunk at this model's latent lengths. segment_block=4 does the
    # same job for the energy variants, matching the reference (lucidrains'
    # train_mac.py runs SEQ_LEN 512 on a 4-token grid). Retrieval reads PRE-write
    # weights, so the writes the model can feel is chunks - 1.
    #
    # swish, NOT serpent. Every parameter of the memory net is a fast weight, so
    # a periodic activation puts its per-feature FREQUENCIES in the test-time
    # update, and the energy rule's sign-like step is well-conditioned on a
    # linear map but not on a frequency. A parameter-free activation keeps the
    # fast-weight set linear, which the Adam rule's scale-invariance argument
    # assumes, and hands the trunk a genuinely different function class from the
    # periodic modules elsewhere. `swish` is torch's SiLU, the paper's
    # activation; the `silu` key is transformers' copy.
    #
    # The learned gates come with standard mode: to_lr / to_momentum / to_decay
    # are theta_t / eta_t / alpha_t from the paper's Eqs. 13-14, all
    # data-dependent, making the step size LEARNED rather than a constant. The
    # forgetting gate is init-biased to retain (_DECAY_GATE_BIAS).
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
    # mag_energy + STITCHED WRITES across linked batch rows. A pass writes over
    # its whole sequence, which at patch_size 8 and block_size 64 is 8-64 latents
    # - enough for grammar, not for a fact - and the trunk cannot afford longer
    # sequences at this model size. The packer already splits long documents
    # across consecutive rows; with `row_continues` published, the memory threads
    # its state along a run of linked rows, so the write span becomes the run's
    # total length while the TRUNK still sees one row. Memory horizon decouples
    # from trunk sequence length. Total work is unchanged (a run of G rows is G
    # batched calls over b/G rows each); the cost is serialization into G
    # sequential memory calls. Read `memory_run_length` x `memory_chunks` for the
    # span actually written.
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
    # mag_energy_stitch + a GATED WRITE. Energy mode has no per-token write
    # gate at all: the learned theta_t head is standard-mode only, so `lr` is
    # 1.0 at every real token and - because the Adam step is sign-like and the
    # step scale is fixed - a boring token writes exactly as hard as a
    # surprising one. What the memory stores converges on a mean over the
    # stream rather than over the part worth storing.
    #
    # `threshold` writes a token whose surprise exceeds `write_gate_ratio` times
    # the causal running mean of the sequence so far, tilted by a slow-over-fast
    # surprise EMA carried across passes. The running mean is what keeps the bar
    # inside a distribution this narrow (CV 0.09, and the gate's whole range
    # lives in a 10% band); the tilt is what lets it CLOSE, since a purely
    # within-sequence bar is scale-invariant. It is the mode that can decline to
    # write ANYTHING: a chunk where nothing clears the bar holds its weights,
    # the "already well-conditioned" case the dense write cannot express. `topk`
    # is the fixed-capacity alternative (expert-choice MoD over the same score)
    # and the arm to run if the variable capacity turns out to be the confound.
    #
    # A surviving token's step is not diluted - _step_scale fixes the per-chunk
    # magnitude - but a chunk where nothing clears the bar contributes zero, and
    # on this profile's 4-token update grid that is common: `memory_write` reads
    # 0.0151 ungated against 0.0101 gated at init, with `memory_gain` unmoved at
    # 0.357. So the arm trades write volume for selectivity rather than holding
    # volume fixed. It is not a compute saving either - the surprise is scored
    # for every token either way. Read `memory_write_share` (how much of the
    # stream got through) beside `memory_write_selectivity` (whether the gate
    # picked on content at all).
    # mag_energy_stitch_gated with the bar tracked to a TARGET SHARE instead of
    # keyed to the score's mean. The mean-relative bar was measured on the run
    # that used it and does not work: once trained the driver is right-skewed
    # (its mean sits at the 82nd percentile) and spans about +-10% around that
    # mean, which is narrower than the tilt's own range - so the tilt alone
    # swept the gate between writing everything and writing nothing. 19% of
    # steps wrote >=99% of the stream, 10% wrote <2%.
    #
    # A causal RANK has no dependence on the score's distribution at all - a
    # token writes when it sits in the top target fraction of the real tokens at
    # or before it, which depends on their order and nothing else. A tracked
    # quantile bar was tried in between and failed the same way: on a stream with
    # a 9% spread and 10% per-pass location drift it gave 0.22 +- 0.25 against a
    # 0.125 target, where the rank gives 0.113 +- 0.017. The tilt then moves the
    # TARGET, bounded in [0, 1], so a noisy tilt cannot swing the gate end to
    # end. Ranking against the prefix rather than the sequence is what keeps it
    # causal.
    #
    # Read `memory_write_share` against `memory_write_target` on the same card:
    # they should track. `memory_write_selectivity` is still the falsifier, and
    # its ceiling is low - on the trained checkpoint the best any selector can
    # do at this capacity is about 1.06, and the gate measured 1.05.
    "mag_energy_stitch_adaptive": dict(
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
    "mag_energy_stitch_gated": dict(
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
    # Stitched writes AND a differentiable update - the only pairing in which a
    # Stitched writes AND a differentiable update - the only pairing in which a
    # longer write span can pay. In energy mode the state handed between rows of
    # a run is DETACHED, so only run-START rows give the memory net gradient at
    # all (||grad W0|| against unstitched: 0.72x at runs of 2, 0.48x at 4, 0.35x
    # at 8). Standard mode keeps a graph through the carried state, so the same
    # sweep reads 0.99x / 0.97x / 0.93x and the span is nearly free. Everything
    # else tracks mag_standard, including chunk_size 4.
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
    # the write lands. The control that separates "a gated nonlinear module at
    # this depth" from "test-time memory". If it matches its live twin, the
    # adaptation contributes nothing measurable and the honest comparison for
    # anything bigger is against a dense of the same size.
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
    "mag_energy_stitch_adaptive": (
        "mag_energy_stitch with the test-time write gated to a TARGET SHARE. A "
        "token writes when its surprise ranks in the top target fraction of the "
        "sequence so far, and a slow-over-fast surprise EMA moves that target "
        "down as the memory's forecasting improves - so a memory already "
        "predicting the stream writes less of it, and at the extreme writes "
        "nothing and holds its weights. A rank rather than a level is what makes "
        "it immune to a surprise distribution that moves as much as it is wide."
    ),
    "mag_energy_stitch_gated": (
        "mag_energy_stitch with the test-time write GATED per token. Energy "
        "mode carries no learned write gate, so every real token writes the "
        "same fixed-magnitude step and the stored association flattens toward a "
        "mean over the stream. Here a token writes only if its own surprise "
        "exceeds a ratio of the sequence's running mean - itself tilted by a "
        "slow-over-fast surprise EMA - so a memory already "
        "forecasting the stream well can decline to write at all - down to a "
        "chunk that writes nothing and holds its weights exactly. Costs one "
        "extra memory-net forward to score, and saves no compute: what it "
        "changes is which associations the write lands on."
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
