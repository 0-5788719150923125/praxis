from typing import Any, Dict, List, Optional, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.activations import ACT2FN
from praxis.blocks import BLOCK_REGISTRY
from praxis.compression import COMPRESSION_REGISTRY
from praxis.controllers import CONTROLLER_REGISTRY
from praxis.experimental.evolution import GenomicBottleneck
from praxis.orchestration import EXPERT_REGISTRY, PraxisExpert, PraxisManagement

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class BaseDecoder(nn.Module):
    """
    A module that wraps decoder operations.
    """

    __version__ = "0.1.0"

    def __init__(self, config: ConfigType) -> None:
        super().__init__()
        self.debug = config.debug
        self.depth = config.depth
        self.checkpoint_every = config.checkpoint_every
        self.num_experts = config.num_experts
        assert (
            self.num_experts >= self.depth
        ), "`num_experts` should be at least as large as `depth`."
        self.controller = CONTROLLER_REGISTRY.get(config.controller_type)(config)
        self.genome = GenomicBottleneck(config) if config.evolve else False
        self.compressor = COMPRESSION_REGISTRY.get(config.compression_type)(config)
        self.manager = False
        self.locals = nn.ModuleList()
        self.remotes: List[nn.Module] = []
        if config.hivemind:
            self.manager = PraxisManagement(config)
            self.remotes = self.manager.active_remote_experts
        if "scatter" in config.meta or config.expert in ["scatter"]:
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

    def post_layer(self, states: Tensor, current_depth: int) -> Tensor:
        """
        Process states after going through a layer.

        Args:
            states: Hidden states from the layer
            current_depth: Current layer depth

        Returns:
            Processed states
        """
        processed_states = states
        if self.genome and current_depth == 4:
            processed_states = self.genome(processed_states)
        return processed_states

    def post_decoding(self, states: Tensor) -> Tensor:
        """
        Process states after going through all layers.

        Args:
            states: Final hidden states

        Returns:
            Normalized or unchanged states
        """
        if self.norm:
            return self.norm(states)
        else:
            return states

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return current prediction accuracies and other metrics.

        Returns:
            Dictionary of metrics
        """
        extras: Dict[str, Any] = {}
        if self.genome:
            extras = {**extras, **self.genome.get_metrics()}
        return {
            "experts": dict(
                local=len(self.locals),
                remote=len(self.remotes),
            ),
            **extras,
        }
