from praxis.heads.forward import ForwardHead
from praxis.heads.harmonic import HarmonicField, HarmonicHead
from praxis.heads.mtp import MTP_REGISTRY, MultiTokenPrediction
from praxis.heads.tied import TiedWeights

HEAD_REGISTRY = dict(
    forward=ForwardHead,
    tied=TiedWeights,
    harmonic=HarmonicHead,
)
