from praxis.heads.crystal import CrystalClassifier, CrystalHead
from praxis.heads.forward import ForwardHead
from praxis.heads.harmonic import HarmonicField, HarmonicHead
from praxis.heads.mtp import MTP_REGISTRY, MultiTokenPrediction
from functools import partial

from praxis.heads.stacked import SequentialHead
from praxis.heads.tied import TiedWeights


def _harmonic_crystal(amp_modulation: str) -> list:
    """Builders for a SequentialHead: the harmonic field (given modulation,
    transform-only so it allocates no dead classifier) feeding the crystal
    distance classifier."""
    return [
        partial(HarmonicHead, amp_modulation=amp_modulation, build_classifier=False),
        CrystalHead,
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
    # Prismatic: the learned harmonic wave refracted through the crystal - the
    # field's amplitude envelope splits features into spectral bands (a prism),
    # crystal refracts them into logits.
    prismatic=partial(SequentialHead, heads=_harmonic_crystal("learned")),
)
