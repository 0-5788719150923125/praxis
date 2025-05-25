from praxis.heads.forward import ForwardHead
from praxis.heads.tied import TiedHead

HEAD_REGISTRY = dict(
    forward=ForwardHead,
    tied=TiedHead,
)
