from praxis.blocks.transformer import PraxisBlock
from praxis.blocks.nano import PraxisNano
from praxis.blocks.conv import PraxisConv

BLOCK_REGISTRY = {"transformer": PraxisBlock, "nano": PraxisNano, "conv": PraxisConv}
