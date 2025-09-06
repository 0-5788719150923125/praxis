from praxis.heads.forward import ForwardHead
from praxis.heads.tied import TiedWeights

HEAD_REGISTRY = dict(
    forward=ForwardHead,
    tied=TiedWeights,
)
