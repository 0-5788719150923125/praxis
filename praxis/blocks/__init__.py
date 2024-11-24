from praxis.blocks.conv import PraxisConv
from praxis.blocks.nano import PraxisNano
from praxis.blocks.recurrent import PraxisRecurrent
from praxis.blocks.transformer import PraxisTransformer

BLOCK_REGISTRY = {
    "transformer": PraxisTransformer,
    "nano": PraxisNano,
    "conv": PraxisConv,
    "recurrent": PraxisRecurrent,
}
