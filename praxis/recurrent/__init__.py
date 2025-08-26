from praxis.recurrent.min_gru import MinGRU
from praxis.recurrent.gru import GRU

RECURRENT_REGISTRY = dict(
    min_gru=MinGRU,
    gru=GRU,
)
