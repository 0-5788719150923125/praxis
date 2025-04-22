import torch
import torch.nn as nn
from torch import Tensor

from praxis.orchestration.hivemind import P2PDaemonError, P2PHandlerError


def create_forward(
    expert: nn.Module,
    stack: nn.Module,
    hidden_states: Tensor,
    attention_mask: Tensor,
    past_key_values: Tensor,
    current_state: Tensor,
    current_depth: int,
    block_ids: Tensor,
    should_checkpoint: bool = False,
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
        if stack.behavior:
            hidden_states, attention_mask = stack.behavior.add_context(
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
        if stack.behavior:
            states, attention_mask = stack.behavior.remove_context(
                states, attention_mask
            )

        return states, layer_kv, state_update, aux_loss

    try:
        if should_checkpoint:
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
    except (P2PDaemonError, P2PHandlerError) as e:
        if stack.debug:
            print(e)
        stack.manager.handle_failure(experts[current_depth])
        return None


def should_checkpoint(training: bool, current_depth: int, checkpoint_every: int):
    return (
        training
        and current_depth != 0
        and checkpoint_every is not None
        and checkpoint_every != 0
        and current_depth % checkpoint_every
    )
