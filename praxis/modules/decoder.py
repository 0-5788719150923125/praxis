import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from praxis.blocks import BLOCK_REGISTRY
from praxis.modules.controller import PraxisController
from praxis.modules.experts import PraxisExpert
from praxis.modules.memory import PraxisMemory
from praxis.modules.router import PraxisMixtureOfDepths
from praxis.orchestration.hivemind import (
    P2PDaemonError,
    P2PHandlerError,
    PraxisManagement,
)


class PraxisDecoder(nn.Module):
    """
    A module that wraps the entire transformer decoder block.
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.debug = config.debug
        self.depth = config.depth
        self.sparse = config.sparse
        self.shuffle = config.shuffle
        self.memory = False
        if config.memory:
            self.memory = PraxisMemory(config)
        self.manager = False
        self.remote_experts = []
        if config.hivemind:
            self.manager = PraxisManagement(config)
            self.remote_experts = self.manager.active_remote_experts
        self.local_experts = nn.ModuleList()
        for i in range(config.num_experts):
            if self.manager:
                block = self.manager.register_expert(config)
            else:
                block = BLOCK_REGISTRY[config.block_type](config)
            router = False
            if config.sparse and i % 2 != 0:
                router = PraxisMixtureOfDepths(config)
            expert = PraxisExpert(
                config, block=block, memory=self.memory, router=router
            )
            self.local_experts.append(expert)
        if self.manager:
            self.manager.serve_experts()
        self.navigator = False
        if config.autopilot:
            self.navigator = PraxisController(config, len(self.local_experts) * 3)
        self._define_checkpoints(config.memory_profile, self.depth)

    def forward(self, inputs: Tensor, attention_mask: Tensor):
        experts = list(self.local_experts) + list(self.remote_experts)
        original_order = experts.copy()
        if self.shuffle:
            random.shuffle(experts)

        hidden_states = inputs
        aux_losses = []

        first_expert_idx = original_order.index(experts[0])
        route = [str(first_expert_idx)]

        next_expert_idx = None
        for i in range(self.depth):
            try:
                expert = experts[i]
                if not self.training and next_expert_idx is not None:
                    expert = experts[next_expert_idx]
                    route.append(str(next_expert_idx))

                new_states, aux_loss = self._create_forward(
                    expert, hidden_states, attention_mask, i
                )
                aux_losses.append(aux_loss)

                if self.navigator:
                    # Predict the optimal next-expert index
                    aux_loss, next_expert_idx = self.navigator(
                        original_order, experts, expert, new_states
                    )
                    aux_losses.append(aux_loss)

                # Commit to self
                hidden_states = new_states

            except (P2PDaemonError, P2PHandlerError) as e:
                # Prune dead peers
                if self.manager:
                    if self.debug:
                        print(e)
                    self.manager.handle_failure(expert)
                    continue

        if self.debug and not self.training and self.navigator:
            print(f"DEBUG: routing through: {' -> '.join(route)}")

        self.get_metrics()

        return hidden_states, sum(aux_losses)

    def _define_checkpoints(self, strategy="speed", num_layers=0):
        self.checkpoint_indices = []  # speed / no gradient checkpointing
        if strategy == "aggressive":
            # every layer
            self.checkpoint_indices = [i for i in range(num_layers)]
        elif strategy == "balanced":
            # every fourth layer
            self.checkpoint_indices = [i for i in range(num_layers) if i % 4 == 0]

    def _create_forward(
        self,
        expert: nn.Module,
        hidden_states: Tensor,
        attention_mask: Tensor,
        current_depth: int,
    ):
        def custom_forward(hidden_states, attention_mask, current_depth):
            return expert(hidden_states, attention_mask, current_depth)

        do_checkpoint = True if current_depth in self.checkpoint_indices else False
        if do_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                current_depth,
                use_reentrant=False,
            )
        else:
            return custom_forward(hidden_states, attention_mask, current_depth)

    def get_metrics(self):
        """Return current prediction accuracies"""
        if self.memory:
            return {**self.memory.get_metrics()}
        # if self.navigator:
        #     return {
        #         "mean": self.navigator.get_mean_accuracy(),
        #         "per_expert": self.navigator.get_all_accuracies(),
        #     }
        return {}
