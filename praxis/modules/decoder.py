import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.orchestration.hivemind import P2PDaemonError, P2PHandlerError
from praxis.stacks import PraxisStack


class PraxisDecoder(nn.Module):
    """
    A module that wraps the entire transformer decoder block.
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig"):
        super().__init__()
        self.debug = config.debug
        self.stack = PraxisStack(config)
        self.manager = self.stack.manager
        self._define_checkpoints(config.strategy, self.stack.depth)

    def forward(self, inputs: Tensor, current_state: Tensor, attention_mask: Tensor):
        experts = list(self.stack.locals) + list(self.stack.remotes)
        original_order = experts.copy()
        if hasattr(self.stack.behavior, "shuffle_experts"):
            experts = self.stack.behavior.shuffle_experts(experts)

        hidden_states = inputs
        aux_losses = []

        next_expert_idx = None
        for i in range(self.stack.depth):
            try:
                expert = experts[i]
                if next_expert_idx is not None:
                    expert = experts[next_expert_idx]

                new_states, aux_loss = self._create_forward(
                    expert, hidden_states, current_state, attention_mask, i
                )
                aux_losses.append(aux_loss)

                if hasattr(self.stack.behavior, "get_next_expert"):
                    # Predict the optimal next-expert index
                    aux_loss, next_expert_idx = self.stack.behavior.get_next_expert(
                        new_states, i, original_order, experts, expert
                    )
                    aux_losses.append(aux_loss)

                if hasattr(self.stack, "post_compute"):
                    new_states = self.stack.post_compute(new_states, i)

                # Commit to self
                hidden_states = new_states

            except (P2PDaemonError, P2PHandlerError) as e:
                # Prune dead peers
                if self.debug:
                    print(e)
                self.manager.handle_failure(expert)
                continue

        if hasattr(self.stack, "post_decoding"):
            hidden_states = self.stack.post_decoding(hidden_states)

        if hasattr(self.stack.behavior, "reset_route"):
            self.stack.behavior.reset_route()

        return hidden_states, current_state, sum(aux_losses)

    def _create_forward(
        self,
        expert: nn.Module,
        hidden_states: Tensor,
        current_state: Tensor,
        attention_mask: Tensor,
        current_depth: int,
    ):
        def custom_forward(hidden_states, current_state, attention_mask, current_depth):
            if self.stack.behavior:
                # Add positional context to both hidden states and attention mask
                hidden_states, attention_mask = self.stack.behavior.add_context(
                    hidden_states, attention_mask, current_depth
                )
                # Forward pass
                states, state_update, aux_loss = expert(
                    hidden_states,
                    current_state[current_depth],
                    attention_mask,
                    current_depth,
                )
                # Remove context from both hidden states and attention mask
                states, attention_mask = self.stack.behavior.remove_context(
                    states, attention_mask
                )
            else:
                states, state_update, aux_loss = expert(
                    hidden_states,
                    current_state[current_depth],
                    attention_mask,
                    current_depth,
                )

            if state_update is not None:
                current_state[current_depth] = state_update

            return states, aux_loss

        if self.training and self._should_checkpoint(current_depth):
            return torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                current_state,
                attention_mask,
                current_depth,
                use_reentrant=False,
            )
        else:
            return custom_forward(
                hidden_states, current_state, attention_mask, current_depth
            )

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
        return self.stack.get_metrics()
