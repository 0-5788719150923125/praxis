from praxis.blocks.transformer import PraxisBlock, EXPERT_REGISTRY, EXPERT_CONFIGS
from praxis.blocks.nano import PraxisNano

BLOCK_REGISTRY = {"default": PraxisBlock, "nano": PraxisNano}
