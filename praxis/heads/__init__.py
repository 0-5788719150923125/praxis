from praxis.heads.crystal import CrystalClassifier, CrystalHead
from praxis.heads.forward import ForwardHead
from praxis.heads.harmonic import HarmonicField, HarmonicHead
from praxis.heads.mtp import MTP_REGISTRY, MultiTokenPrediction
from praxis.heads.stacked import StackedHead
from praxis.heads.tied import TiedWeights

HEAD_REGISTRY = dict(
    forward=ForwardHead,
    tied=TiedWeights,
    harmonic=HarmonicHead,
    crystal=CrystalHead,
    # Harmonic field feeding the crystal classifier (composed, not parallel).
    crystal_harmonic=StackedHead,
)
