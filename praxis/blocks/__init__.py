from praxis.blocks.conv import ConvolutionalBlock
from praxis.blocks.gru import GRUBlock
from praxis.blocks.min import MinGRUBlock
from praxis.blocks.mru import MRUBlock
from praxis.blocks.nano import NanoBlock
from praxis.blocks.recurrent import RecurrentBlock
from praxis.blocks.ssm import SSMBlock
from praxis.blocks.transformer import TransformerBlock

BLOCK_REGISTRY = {
    "conv": ConvolutionalBlock,
    "gru": GRUBlock,
    "mru": MRUBlock,
    "min": MinGRUBlock,
    "nano": NanoBlock,
    "recurrent": RecurrentBlock,
    "ssm": SSMBlock,
    "transformer": TransformerBlock,
}
