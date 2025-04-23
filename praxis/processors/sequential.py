import torch
from torch import nn

from praxis.processors.checkpoint import create_forward, should_checkpoint


class SequentialProcessor(nn.Module):
    def __init__(self, config: "AutoConfig"):
        super().__init__()

    def forward(
        self,
        experts,
        stack,
        inputs,
        attention_mask,
        past_key_values,
        block_ids,
        current_state,
        original_order,
        training: bool = False,
    ):
        hidden_states = inputs
        new_states = []
        aux_losses = []

        for i in range(stack.depth):
            next_expert_idx = i
            if hasattr(stack.behavior, "get_next_expert"):
                aux_loss, next_expert_idx = stack.behavior.get_next_expert(
                    hidden_states, i, original_order, experts
                )
                aux_losses.append(aux_loss)

            expert = experts[next_expert_idx]

            layer_state = current_state[i] if current_state is not None else None
            hidden_states, past_key_values, layer_state, aux_loss = create_forward(
                expert,
                stack,
                hidden_states,
                attention_mask,
                past_key_values,
                layer_state,
                i,
                block_ids,
                should_checkpoint(training, i, stack.checkpoint_every),
            )
            new_states.append(layer_state)
            aux_losses.append(aux_loss)

            if hasattr(stack, "post_layer"):
                hidden_states = stack.post_layer(hidden_states, i)

        if hasattr(stack, "post_decoding"):
            hidden_states = stack.post_decoding(hidden_states)

        if hasattr(stack.behavior, "reset_route"):
            stack.behavior.reset_route()

        return hidden_states, past_key_values, current_state, sum(aux_losses)
