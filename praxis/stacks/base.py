from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.activations import ACT2FN
from praxis.blocks import BLOCK_REGISTRY
from praxis.modules.behaviors import LayerShuffle, MixtureRouter, PraxisGraph
from praxis.modules.evolution import GenomicBottleneck
from praxis.modules.experts import EXPERT_REGISTRY, PraxisExpert
from praxis.modules.router import PraxisMixtureOfDepths
from praxis.orchestration.hivemind import PraxisManagement


class PraxisStack(nn.Module):
    """
    A module that wraps the stack of layers in a decoder.
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig"):
        super().__init__()
        self.depth = config.depth
        self.num_experts = config.num_experts
        assert (
            self.num_experts >= self.depth
        ), "`num_experts` should be at least as large as `depth`."
        self.sparse = config.sparse
        if config.graph:
            self.behavior = PraxisGraph(config)
        elif config.router:
            self.behavior = MixtureRouter(config)
        else:
            self.behavior = LayerShuffle(config) if config.shuffle else False
        self.genome = GenomicBottleneck(config) if config.evolve else False
        self.manager = False
        self.locals = nn.ModuleList()
        self.remotes = []
        if config.hivemind:
            self.manager = PraxisManagement(config)
            self.remotes = self.manager.active_remote_experts
        if "tied" in config.meta or config.expert in ["scatter"]:
            block = BLOCK_REGISTRY[config.block_type](config)
            expert = PraxisExpert(config, block=block)
            for i in range(self.num_experts):
                self.locals.append(expert)
        else:
            for i in range(self.num_experts):
                if self.manager:
                    block = self.manager.register_expert(config)
                else:
                    block = BLOCK_REGISTRY[config.block_type](config)
                expert = PraxisExpert(config, block=block)
                self.locals.append(expert)
        self.norm = (
            nn.LayerNorm(config.hidden_size, bias=True)
            if config.block_type == "mru"
            else False
        )
        if self.manager:
            self.manager.serve_experts()

    def post_compute(self, states, current_depth):
        processed_states = states
        if self.genome and current_depth == 4:
            processed_states = self.genome(processed_states)
        return processed_states

    def post_decoding(self, states):
        if self.norm:
            return self.norm(states)
        else:
            return states

    def get_metrics(self):
        """Return current prediction accuracies"""
        extras = {}
        if self.genome:
            extras = {**extras, **self.genome.get_metrics()}
        return {
            "experts": dict(
                local=len(self.locals),
                remote=len(self.remotes),
            ),
            **extras,
        }
