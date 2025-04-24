import torch
from torch import nn

from praxis.decoders.checkpoint import create_forward, should_checkpoint
from praxis.stacks import PraxisStack


class SequentialDecoder(nn.Module):
    def __init__(self, config: "AutoConfig"):
        super().__init__()
        self.stack = PraxisStack(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        past_key_values,
        current_state,
        block_ids,
    ):
        new_states = []
        aux_losses = []

        sequential_experts = list(self.stack.locals) + list(self.stack.remotes)
        ordered_experts = self.stack.controller.sort_experts(sequential_experts.copy())

        for i in range(self.stack.depth):
            aux_loss, next_expert_idx = self.stack.controller.get_next_expert(
                hidden_states,
                current_depth=i,
                original_experts=sequential_experts,
                current_experts=ordered_experts,
            )
            aux_losses.append(aux_loss)
            if next_expert_idx is None:
                break

            expert = ordered_experts[next_expert_idx]

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

            hidden_states = self.stack.post_layer(hidden_states, i)

        hidden_states = self.stack.post_decoding(hidden_states)

        self.stack.controller.reset_route(hidden_states)

        return hidden_states, past_key_values, current_state, sum(aux_losses)
