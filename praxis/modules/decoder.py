import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from praxis.modules.controller import PraxisController
from praxis.modules.experts import PraxisExpert
from praxis.orchestration.hivemind import (
    P2PDaemonError,
    P2PHandlerError,
    PraxisManagement,
)


class PraxisDecoder(nn.Module):
    """
    A module that wraps the entire decoder block (and all intermediate layers)
    in a single class.
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.debug = config.debug
        self.depth = config.depth
        self.sparse = config.sparse
        self.shuffle = config.shuffle
        self.random = random.Random(config.seed)
        self.manager = False
        self.remote_experts = []
        if config.hivemind:
            self.manager = PraxisManagement(config)
            self.remote_experts = self.manager.active_remote_experts
        self.local_experts = nn.ModuleList(
            [PraxisExpert(config, self.manager) for _ in range(config.num_experts)]
        )
        self.use_autopilot = config.autopilot
        if self.use_autopilot:
            self.navigator = PraxisController(config, len(self.local_experts) * 3)
        self._define_checkpoints(config.memory_profile, self.depth)
        if self.manager:
            self.manager.serve_experts()

    def forward(self, inputs: Tensor, attention_mask: Tensor):
        experts = list(self.local_experts) + list(self.remote_experts)
        original_order = experts.copy()
        if self.shuffle:
            self.random.shuffle(experts)

        hidden_states = inputs
        aux_losses = []
        next_expert_idx = None

        first_expert_idx = original_order.index(experts[0])
        route = [str(first_expert_idx)]

        for i in range(self.depth):
            use_router = True if self.sparse and i % 2 != 0 else False
            gradient_checkpointing = True if i in self.checkpoint_indices else False
            try:
                expert = experts[i]
                if not self.training and next_expert_idx is not None:
                    expert = experts[next_expert_idx]
                    route.append(str(next_expert_idx))

                new_states, aux_loss = self._create_forward(
                    expert,
                    hidden_states,
                    attention_mask,
                    use_router,
                    gradient_checkpointing,
                )

                aux_losses.append(aux_loss)

                # Predict the "true" index of each expert
                if self.use_autopilot:
                    aux_loss, next_expert_idx, should_exit = self.navigator(
                        experts, expert, new_states, i
                    )
                    aux_losses.append(aux_loss)
                    if should_exit:
                        break

                # Commit to self
                hidden_states = new_states

            except (P2PDaemonError, P2PHandlerError) as e:
                # Prune dead peers
                if self.manager:
                    if self.debug:
                        print(e)
                    self.manager.handle_failure(expert)
                    continue

        if self.use_autopilot:
            hidden_states = self.navigator.merge_states(hidden_states)

        if self.debug and not self.training and self.use_autopilot:
            print(f"DEBUG: routing through: {' -> '.join(route)}")

        return hidden_states, sum(aux_losses)

    def get_prediction_accuracies(self):
        """Return current prediction accuracies"""
        if self.use_autopilot:
            return {
                "mean": self.navigator.get_mean_accuracy(),
                "per_expert": self.navigator.get_all_accuracies(),
            }
        return None

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
        use_router: bool,
        gradient_checkpointing=False,
    ):
        def custom_forward(hidden_states, attention_mask, use_router):
            return expert(hidden_states, attention_mask, use_router)

        if gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                use_router,
                use_reentrant=False,
            )
        else:
            return custom_forward(hidden_states, attention_mask, use_router)
