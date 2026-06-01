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


HEAD_REGISTRY = dict(
    forward=ForwardHead,
    tied=TiedWeights,
    harmonic=HarmonicHead,
    crystal=CrystalHead,
    # Harmonic field feeding the crystal classifier, composed dynamically by
    # SequentialHead: bare grid (off) or a fixed single oscillation (static).
    crystal_harmonic=partial(SequentialHead, heads=_harmonic_crystal("off")),
    crystal_harmonic_static=partial(SequentialHead, heads=_harmonic_crystal("static")),
    # Prismatic: a top-level parallel split balancing bias against variance per
    # token. Branch 0 is a harmonic field read out by a plain linear head (a
    # strong structural prior); branch 1 refracts a second field through the
    # crystal distance classifier (the more expressive arm). A learned per-token
    # gate weights the two logit streams:
    #   Parallel(Sequential(HarmonicField), Sequential(HarmonicField, CrystalClassifier))
    prismatic=partial(
        ParallelHead,
        branches=[
            partial(SequentialHead, heads=[_field("learned", build_classifier=True)]),
            partial(SequentialHead, heads=[_field("learned"), CrystalHead]),
        ],
    ),
)
