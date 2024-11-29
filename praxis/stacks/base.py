import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from praxis.activations import ACT2FN
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
        self.sparse = config.sparse
        self.shuffle = config.shuffle
        if not self.shuffle:
            assert (
                self.num_experts == self.depth
            ), "There is no point in making `num_experts` greater than or less than `depth`, when `shuffle != True`. The additional experts would never be used."
        self.behavior = LayerShuffle(config) if self.shuffle else False
        self.genome = GenomicBottleneck(config) if config.evolve else False
        self.manager = False
        self.remotes = []
        if config.hivemind:
            self.manager = PraxisManagement(config)
            self.remotes = self.manager.active_remote_experts
        self.locals = nn.ModuleList()
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
        self.navigator = False
        if config.autopilot:
            self.navigator = PraxisController(config, len(self.locals) * 3)


class LayerShuffle(nn.Module):
    """
    This module implements a basic form of LayerShuffle-Position, though we use it
    as a differentiable "context token" and input-manipulation/preparation mechanism,
    rather than a positional encoder.
    https://arxiv.org/abs/2407.04513
    """

    def __init__(self, config: AutoConfig, num_context_tokens: int = 3):
        super().__init__()
        self.num_context_tokens = num_context_tokens
        self.embeddings = nn.Parameter(
            torch.randn(config.depth, num_context_tokens, config.num_dims)
        )
        nn.init.normal_(self.embeddings, mean=0.0, std=0.02)
        # Add mixing components
        bottleneck = config.num_dims // 4
        self.mixer = nn.Sequential(
            nn.Linear(config.num_dims, bottleneck),
            ACT2FN["gelu"],
            nn.Dropout(config.dropout),
            nn.Linear(bottleneck, config.num_dims),
        )
        # Gate for balancing position vs content influence
        self.gate = nn.Linear(config.num_dims, 1)

    def add_context(self, hidden_states: Tensor, position: int) -> Tensor:
        # Get position-based embeddings
        pos_embeds = self.embeddings[position]  # [num_tokens, dims]

        # Create content-based context
        # Average the hidden states for content representation
        content = hidden_states.mean(dim=1)  # [batch, dims]
        content_context = self.mixer(content)  # [batch, dims]

        # Expand position embeddings
        pos_embeds = pos_embeds.expand(hidden_states.shape[0], -1, -1)
        # Expand content context
        content_context = content_context.unsqueeze(1).expand(
            -1, self.num_context_tokens, -1
        )

        # Compute mixing ratio
        gate = torch.sigmoid(self.gate(content_context))

        # Mix position and content information
        mixed_context = gate * pos_embeds + (1 - gate) * content_context

        return torch.cat([mixed_context, hidden_states], dim=1)

    def remove_context(self, hidden_states: Tensor) -> Tensor:
        return hidden_states[:, self.num_context_tokens :, :]

    def shuffle_experts(self, experts: list, allow_resampling: bool = False) -> list:
        depth = self.embeddings.shape[0]
        if allow_resampling:
            return random.choices(experts, k=depth)
        else:
            return random.sample(experts, k=depth)
