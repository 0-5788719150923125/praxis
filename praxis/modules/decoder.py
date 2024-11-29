import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

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
        self.manager = self.stack.manager
        self._define_checkpoints(config.strategy, self.stack.depth)

    def forward(self, inputs: Tensor, attention_mask: Tensor):
        experts = list(self.stack.locals) + list(self.stack.remotes)
        original_order = experts.copy()
        if self.stack.behavior:
            experts = self.stack.behavior.shuffle_experts(experts)

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

                if self.stack.genome and i == 4:
                    new_states = self.stack.genome(new_states)

                # Commit to self
                hidden_states = new_states

            except (P2PDaemonError, P2PHandlerError) as e:
                # Prune dead peers
                if self.debug:
                    print(e)
                self.manager.handle_failure(expert)
                continue

        if self.debug and not self.training and self.stack.navigator:
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
            if self.stack.behavior:
                # Add positional context
                hidden_states = self.stack.behavior.add_context(
                    hidden_states, current_depth
                )
                # Adjust attention mask to account for extra token
                context_mask = attention_mask.new_ones(attention_mask.shape[0], 1)
                attention_mask = torch.cat([context_mask, attention_mask], dim=1)
                # Forward pass
                states, aux_loss = expert(hidden_states, attention_mask, current_depth)
                # Remove context token
                states = self.stack.behavior.remove_context(states)
                return states, aux_loss
            else:
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
        extras = {}
        if self.stack.genome:
            extras = {**extras, **self.stack.genome.get_metrics()}
        return {
            "experts": dict(
                local=len(self.stack.locals),
                remote=len(self.stack.remotes),
            ),
            **extras,
        }
