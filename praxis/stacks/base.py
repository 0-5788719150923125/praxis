import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from praxis.blocks import BLOCK_REGISTRY
from praxis.modules.controller import PraxisController
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
        self.shuffle = config.shuffle
        if not self.shuffle:
            assert (
                self.num_experts == self.depth
            ), "There is no point in making `num_experts` greater than or less than `depth`, when `shuffle != True`. The additional experts would never be used."
        self.permutations = False
        if self.shuffle:
            self.permutations = LayerShuffle(config)
        self.sparse = config.sparse
        self.genome = GenomicBottleneck(config) if config.evolve else False
        self.manager = False
        self.remote_experts = []
        if config.hivemind:
            self.manager = PraxisManagement(config)
            self.remote_experts = self.manager.active_remote_experts
        self.local_experts = nn.ModuleList()
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
                self.local_experts.append(expert)
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
                self.local_experts.append(expert)
        if self.manager:
            self.manager.serve_experts()
        self.navigator = False
        if config.autopilot:
            self.navigator = PraxisController(config, len(self.local_experts) * 3)


class LayerShuffle(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(config.depth, config.num_dims))
        # Initialize with small values
        nn.init.normal_(self.embeddings, mean=0.0, std=0.02)

    def add_context(self, hidden_states: Tensor, position: int) -> Tensor:
        # Get position embedding for current layer
        pos_embed = self.embeddings[position]
        # Add as a new "token" at the start
        # Expand pos_embed to match batch dimension
        pos_embed = pos_embed.expand(hidden_states.shape[0], -1)
        # Add sequence dimension
        pos_embed = pos_embed.unsqueeze(1)
        return torch.cat([pos_embed, hidden_states], dim=1)

    def remove_context(self, hidden_states: Tensor) -> Tensor:
        # Remove the first "token" (our added context)
        return hidden_states[:, 1:, :]

    def shuffle_experts(self, experts: list) -> list:
        """Returns a new shuffled list of experts without modifying original"""
        return random.sample(experts, len(experts))
