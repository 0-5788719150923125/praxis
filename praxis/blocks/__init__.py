from praxis.blocks.conv import PraxisConv
from praxis.blocks.min import PraxisGRU
from praxis.blocks.mru import PraxisMRU
from praxis.blocks.nano import PraxisNano
from praxis.blocks.smear import PraxisSMEAR
from praxis.blocks.transformer import PraxisTransformer

BLOCK_REGISTRY = {
    "conv": PraxisConv,
    "mru": PraxisMRU,
    "min": PraxisGRU,
    "nano": PraxisNano,
    "recurrent": PraxisSMEAR,
    "transformer": PraxisTransformer,
}
