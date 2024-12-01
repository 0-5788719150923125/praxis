from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from praxis.activations import ACT2FN
from praxis.blocks import BLOCK_REGISTRY
from praxis.modules.behaviors import LayerShuffle, PraxisGraph
from praxis.modules.evolution import GenomicBottleneck
from praxis.modules.experts import EXPERT_REGISTRY, PraxisExpert
from praxis.modules.router import PraxisMixtureOfDepths
from praxis.orchestration.hivemind import PraxisManagement


class PraxisStack(nn.Module):
    """
    A module that wraps the stack of layers in a decoder.
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.depth = config.depth
        self.num_experts = config.num_experts
        assert (
            self.num_experts >= self.depth
        ), "`num_experts` should be at least as large as `depth`."
        self.sparse = config.sparse
        if config.graph:
            self.behavior = PraxisGraph(config)
        else:
            self.behavior = (
                LayerShuffle(config, num_context_tokens=3) if config.shuffle else False
            )
        self.genome = GenomicBottleneck(config) if config.evolve else False
        self.manager = False
        self.locals = nn.ModuleList()
        self.remotes = []
        if config.hivemind:
            self.manager = PraxisManagement(config)
            self.remotes = self.manager.active_remote_experts
        if config.block_type == "recurrent":
            blocks = [
                EXPERT_REGISTRY["recurrent"](config) for _ in range(self.num_experts)
            ]
            for i in range(self.num_experts):
                mixture = BLOCK_REGISTRY["recurrent"](config, blocks)
                router = False
                use_router = config.sparse and i % 2 != 0
                if use_router:
                    router = PraxisMixtureOfDepths(config)
                expert = PraxisExpert(config, block=mixture, router=router)
                self.locals.append(expert)
        else:
            for i in range(self.num_experts):
                if self.manager:
                    block = self.manager.register_expert(config)
                else:
                    block = BLOCK_REGISTRY[config.block_type](config)
                router = False
                if "chaos" in config.meta:
                    use_router = config.sparse
                elif "thin" in config.meta:
                    use_router = config.sparse and i % 4 != 0
                else:
                    use_router = config.sparse and i % 2 != 0
                if use_router:
                    router = PraxisMixtureOfDepths(config)
                expert = PraxisExpert(config, block=block, router=router)
                self.locals.append(expert)
        if self.manager:
            self.manager.serve_experts()
