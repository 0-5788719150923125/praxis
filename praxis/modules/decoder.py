import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from praxis.modules.evolution import GenomicBottleneck
from praxis.orchestration.hivemind import P2PDaemonError, P2PHandlerError
from praxis.stacks import PraxisStack


class PraxisDecoder(nn.Module):
    """
    A module that wraps the entire transformer decoder block.
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.debug = config.debug
        self.stack = PraxisStack(config)
        self.genome = GenomicBottleneck(config) if config.evolve else False
        self.manager = self.stack.manager
        self._define_checkpoints(config.strategy, self.stack.depth)

    def forward(self, inputs: Tensor, attention_mask: Tensor):
        experts = list(self.stack.local_experts) + list(self.stack.remote_experts)
        original_order = experts.copy()
        if self.stack.shuffle:
            random.shuffle(experts)

        hidden_states = inputs
        aux_losses = []

        route = []

        next_expert_idx = None
        for i in range(self.stack.depth):
            try:
                expert = experts[i]
                if not self.training and next_expert_idx is not None:
                    expert = experts[next_expert_idx]
                    route.append(str(next_expert_idx))
                elif not self.training and self.stack.shuffle:
                    route.append(str(original_order.index(experts[i])))

                new_states, aux_loss = self._create_forward(
                    expert, hidden_states, attention_mask, i
                )
                aux_losses.append(aux_loss)

                if self.stack.navigator:
                    # Predict the optimal next-expert index
                    aux_loss, next_expert_idx = self.navigator(
                        original_order, experts, expert, new_states
                    )
                    aux_losses.append(aux_loss)

                if self.genome and i == 4:
                    new_states = self.genome(new_states)

                # Commit to self
                hidden_states = new_states

            except (P2PDaemonError, P2PHandlerError) as e:
                # Prune dead peers
                if self.debug:
                    print(e)
                self.manager.handle_failure(expert)
                continue

        if self.debug and not self.training and self.stack.shuffle:
            print(f"DEBUG: routing through: {' -> '.join(route)}")

        return hidden_states, sum(aux_losses)

    def _create_forward(
        self,
        expert: nn.Module,
        hidden_states: Tensor,
        attention_mask: Tensor,
        current_depth: int,
    ):
        def custom_forward(hidden_states, attention_mask, current_depth):
            return expert(hidden_states, attention_mask, current_depth)

        if self.training and self._should_checkpoint(current_depth):
            return torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                current_depth,
                use_reentrant=False,
            )
        else:
            return custom_forward(hidden_states, attention_mask, current_depth)

    def _define_checkpoints(self, strategy="speed", num_layers=0):
        self.checkpoint_indices = []  # speed / no gradient checkpointing
        if strategy == "aggressive":
            # every layer
            self.checkpoint_indices = [i for i in range(num_layers)]
        elif strategy == "balanced":
            # every fourth layer
            self.checkpoint_indices = [i for i in range(num_layers) if i % 4 == 0]

    def _should_checkpoint(self, layer_idx):
        return True if layer_idx in self.checkpoint_indices else False

    def get_metrics(self):
        """Return current prediction accuracies"""
        return {
            "experts": dict(
                local=len(self.stack.local_experts),
                remote=len(self.stack.remote_experts),
            )
        }
