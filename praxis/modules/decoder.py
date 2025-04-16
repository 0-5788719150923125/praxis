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
        self.checkpoint_every = config.checkpoint_every

    def forward(
        self,
        inputs: Tensor,
        current_state: Tensor,
        attention_mask: Tensor,
        past_key_values=None,
        block_ids=None,
    ):

        experts = list(self.stack.locals) + list(self.stack.remotes)
        original_order = experts.copy()
        if hasattr(self.stack.behavior, "shuffle_experts"):
            experts = self.stack.behavior.shuffle_experts(experts)

        hidden_states = inputs
        new_states = []
        aux_losses = []

        next_expert_idx = None
        for i in range(self.stack.depth):
            try:
                expert = experts[i]
                if next_expert_idx is not None:
                    expert = experts[next_expert_idx]

                layer_state = current_state[i] if current_state is not None else None
                hidden_update, past_key_values, layer_state, aux_loss = (
                    self._create_forward(
                        expert,
                        hidden_states,
                        attention_mask,
                        past_key_values,
                        layer_state,
                        i,
                        block_ids,
                    )
                )
                new_states.append(layer_state)
                aux_losses.append(aux_loss)

                if hasattr(self.stack.behavior, "get_next_expert"):
                    # Predict the optimal next-expert index
                    aux_loss, next_expert_idx = self.stack.behavior.get_next_expert(
                        hidden_update, i, original_order, experts, expert
                    )
                    aux_losses.append(aux_loss)

                if hasattr(self.stack, "post_compute"):
                    hidden_update = self.stack.post_compute(hidden_update, i)

                # Commit to self
                hidden_states = hidden_update

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

        return hidden_states, past_key_values, current_state, sum(aux_losses)

    def _create_forward(
        self,
        expert: nn.Module,
        hidden_states: Tensor,
        attention_mask: Tensor,
        past_key_values: Tensor,
        current_state: Tensor,
        current_depth: int,
        block_ids: Tensor,
    ):
        def custom_forward(
            hidden_states,
            attention_mask,
            past_key_values,
            current_state,
            current_depth,
            block_ids,
        ):
            # Add positional context to both hidden states and attention mask
            if self.stack.behavior:
                hidden_states, attention_mask = self.stack.behavior.add_context(
                    hidden_states, attention_mask, current_depth
                )
            # Forward pass
            states, layer_kv, state_update, aux_loss = expert(
                hidden_states,
                attention_mask,
                past_key_values,
                current_state,
                current_depth,
                block_ids,
            )
            # Remove context from both hidden states and attention mask
            if self.stack.behavior:
                states, attention_mask = self.stack.behavior.remove_context(
                    states, attention_mask
                )

            return states, layer_kv, state_update, aux_loss

        should_checkpoint = (
            current_depth != 0
            and self.checkpoint_every is not None
            and current_depth % self.checkpoint_every
        )
        if self.training and should_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                past_key_values,
                current_state,
                current_depth,
                block_ids,
                use_reentrant=False,
            )
        else:
            return custom_forward(
                hidden_states,
                attention_mask,
                past_key_values,
                current_state,
                current_depth,
                block_ids,
            )

    def get_metrics(self):
        """Return current prediction accuracies"""
        return self.stack.get_metrics()
