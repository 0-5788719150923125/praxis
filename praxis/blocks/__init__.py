from praxis.blocks.conv import ConvolutionalBlock
from praxis.blocks.min import MinGRUBlock
from praxis.blocks.mru import MRUBlock
from praxis.blocks.nano import NanoBlock
from praxis.blocks.recurrent import RecurrentBlock
from praxis.blocks.transformer import TransformerBlock

BLOCK_REGISTRY = {
    "conv": ConvolutionalBlock,
    "mru": MRUBlock,
    "min": MinGRUBlock,
    "nano": NanoBlock,
    "recurrent": RecurrentBlock,
    "transformer": TransformerBlock,
}
