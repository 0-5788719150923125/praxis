from praxis.blocks.conv import PraxisConv
from praxis.blocks.nano import PraxisNano
from praxis.blocks.smear import PraxisSMEAR
from praxis.blocks.transformer import PraxisTransformer

BLOCK_REGISTRY = {
    "transformer": PraxisTransformer,
    "nano": PraxisNano,
    "conv": PraxisConv,
    "recurrent": PraxisSMEAR,
}
