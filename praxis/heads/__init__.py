from praxis.heads.bidirectional import BidirectionalHead
from praxis.heads.forward import ForwardHead

HEAD_REGISTRY = dict(
    forward=ForwardHead,
    bidirectional=BidirectionalHead,
)