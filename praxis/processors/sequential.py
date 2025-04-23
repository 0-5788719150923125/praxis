import torch
from torch import nn

from praxis.processors.checkpoint import create_forward, should_checkpoint
from praxis.stacks import PraxisStack


class SequentialProcessor(nn.Module):
    def __init__(self, config: "AutoConfig"):
        super().__init__()
        self.stack = PraxisStack(config)

    def forward(
        self,
        inputs,
        attention_mask,
        past_key_values,
        block_ids,
        current_state,
    ):
        hidden_states = inputs
        new_states = []
        aux_losses = []

        experts = list(self.stack.locals) + list(self.stack.remotes)
        original_order = experts.copy()
        if hasattr(self.stack.behavior, "shuffle_experts"):
            experts = self.stack.behavior.shuffle_experts(experts)

        for i in range(self.stack.depth):
            next_expert_idx = i
            if hasattr(self.stack.behavior, "get_next_expert"):
                aux_loss, next_expert_idx = self.stack.behavior.get_next_expert(
                    hidden_states, i, original_order, experts
                )
                aux_losses.append(aux_loss)
                if next_expert_idx is None:
                    break

            expert = experts[next_expert_idx]

            layer_state = current_state[i] if current_state is not None else None
            hidden_states, past_key_values, layer_state, aux_loss = create_forward(
                expert,
                self.stack,
                hidden_states,
                attention_mask,
                past_key_values,
                layer_state,
                i,
                block_ids,
                should_checkpoint(self.training, i, self.stack.checkpoint_every),
            )
            new_states.append(layer_state)
            aux_losses.append(aux_loss)

            if hasattr(self.stack, "post_layer"):
                hidden_states = self.stack.post_layer(hidden_states, i)

        if hasattr(self.stack, "post_decoding"):
            hidden_states = self.stack.post_decoding(hidden_states)

        if hasattr(self.stack.behavior, "reset_route"):
            self.stack.behavior.reset_route()

        return hidden_states, past_key_values, current_state, sum(aux_losses)
