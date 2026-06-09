from praxis.heads.crystal import CrystalClassifier, CrystalHead
from praxis.heads.forward import ForwardHead
from praxis.heads.harmonic import HarmonicField, HarmonicHead
from praxis.heads.mtp import MTP_REGISTRY, MultiTokenPrediction
from functools import partial

from praxis.heads.parallel import ParallelHead
from praxis.heads.stacked import SequentialHead
from praxis.heads.tied import TiedWeights


def _field(amp_modulation: str, build_classifier: bool = False):
    """A harmonic field builder; transform-only by default (no dead classifier),
    or terminal (its own linear readout) when ``build_classifier`` is set."""
    return partial(
        HarmonicHead,
        amp_modulation=amp_modulation,
        build_classifier=build_classifier,
    )


def _harmonic_crystal(amp_modulation: str) -> list:
    """Builders for a SequentialHead: the harmonic field (given modulation,
    transform-only so it allocates no dead classifier) feeding the crystal
    distance classifier."""
    return [_field(amp_modulation), CrystalHead]


def _prismatic3_branches() -> list:
    """The three prismatic3 arms: bias (learned field), variance (input field ->
    crystal), and a pure variance-only field. Shared by prismatic3 variants."""
    return [
        partial(SequentialHead, heads=[_field("learned", build_classifier=True)]),
        partial(SequentialHead, heads=[_field("input"), CrystalHead]),
        partial(SequentialHead, heads=[_field("pure", build_classifier=True)]),
    ]


HEAD_REGISTRY = dict(
    forward=ForwardHead,
    tied=TiedWeights,
    harmonic=HarmonicHead,
    crystal=CrystalHead,
    # Harmonic field feeding the crystal classifier, composed dynamically by
    # SequentialHead: bare grid (off) or a fixed single oscillation (static).
    crystal_harmonic=partial(SequentialHead, heads=_harmonic_crystal("off")),
    crystal_harmonic_static=partial(SequentialHead, heads=_harmonic_crystal("static")),
    # Prismatic: a top-level parallel split that makes the bias/variance axes two
    # physical branches. Branch 0 is a harmonic field (learned but static
    # envelope) read out by a plain linear head - the bias arm, a strong
    # structural prior. Branch 1 refracts an input-conditional field (its
    # envelope carries a per-sequence delta, identity at init) through the
    # crystal distance classifier - the variance arm, the expressive one. A
    # learned per-token gate weights the two logit streams, routing features to
    # whichever arm explains them. Each arm emits its own Bias/Variance Strands
    # card (#0 stays collapsed = bias; #1 separates as variance is learned):
    #   Parallel(Sequential(HarmonicField), Sequential(HarmonicField, CrystalClassifier))
    prismatic=partial(
        ParallelHead,
        branches=[
            partial(SequentialHead, heads=[_field("learned", build_classifier=True)]),
            partial(SequentialHead, heads=[_field("input"), CrystalHead]),
        ],
    ),
    # Prismatic + a third, variance-only arm: a "pure" field (no static
    # spectrum; the conditional delta alone, zero at init) with its own linear
    # readout - the mirror of the bias arm. Variance can arrive in bins the
    # bias arms never occupy; its strand card starts empty and grows pure red.
    prismatic3=partial(ParallelHead, branches=_prismatic3_branches()),
    # prismatic3 + level-repulsion on the gate: a pairwise log-gap penalty pushes
    # the three arms' mean weights to DISTINCT tiers (e.g. 70/20/10) and punishes
    # near-ties (70/15/15) where two arms become equally important. Watch the
    # "Parallel Gate Min Gap" card. Strength is baked here, not a config flag.
    prismatic3_repel=partial(
        ParallelHead, branches=_prismatic3_branches(), gate_repulsion=0.02
    ),
)
