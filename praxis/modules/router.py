import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.utils import (
    generate_alternating_values,
    generate_decay_values,
    generate_u_shape_values,
)

MOD_LAYOUT = {
    "decayed": lambda config: generate_decay_values(
        config.depth, reverse=True, center=0.5, lower_bound=config.capacity
    ),
    "ramped": lambda config: generate_decay_values(
        config.depth, center=0.5, lower_bound=config.capacity
    ),
    "u": lambda config: generate_u_shape_values(
        config.depth,
        decay_point=0.2,
        ramp_point=0.8,
        lower_bound=config.capacity,
        steepness=2.0,
    ),
    "skip_3": lambda config: generate_alternating_values(
        size=config.depth, interval=2, capacity=config.capacity
    ),
    "skip_2": lambda config: generate_alternating_values(
        size=config.depth, interval=2, capacity=config.capacity
    ),
    "standard": lambda config: generate_alternating_values(
        size=config.depth, interval=1, capacity=config.capacity
    ),
}


class PraxisMixtureOfDepths(nn.Linear):
    """
    This uses expert-choice routing, which was greatly preferred by the
    original authors of this research: https://arxiv.org/abs/2404.02258
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig"):
        super().__init__(in_features=config.hidden_size, out_features=1)
        self.capacities = MOD_LAYOUT.get(config.mod, "standard")(config)
        print(self.capacities)

    def forward(
        self,
        layer: nn.Module,
        inputs: Tensor,
        attention_mask: Tensor,
        past_key_values: Tensor,
        current_state: Tensor,
        current_depth: Tensor,
        block_ids: Tensor,
    ):

        router_loss = 0
        capacity = self.capacities[current_depth]

        b, s, d = inputs.shape
        k = int(s * capacity)

        # if capacity is 1, then we should process all tokens normally
        if capacity == 1:
            layer_outputs, layer_kv, state_update, aux_loss = layer(
                inputs,
                attention_mask,
                past_key_values,
                current_state,
                current_depth,
                block_ids,
            )
            return layer_outputs, layer_kv, state_update, aux_loss

        # if k is 0, then no tokens will be selected and we should skip this layer
        if k == 0:
            return inputs, past_key_values, current_state, router_loss

        # emit scalar weights for each token
        router_logits = F.linear(inputs, self.weight, self.bias)  # -> batch, seq_len, 1

        #  𝑟𝑙> 𝑃𝛽 (R) - equation 1
        token_weights, token_indices = torch.topk(
            router_logits,
            k,
            dim=1,
            sorted=False,
        )

        # Sort indices by position and get the sorting indices
        token_indices, sort_indices = torch.sort(token_indices, dim=1)

        # Re-order the weights to match the sorted indices
        token_weights = torch.gather(token_weights, dim=1, index=sort_indices)

        # compute aux loss, in order to enforce causality in the top-k operation
        router_loss = self.aux_loss(router_logits, token_indices)

        # expand router predictions to match input dimensions
        indices_expanded = token_indices.expand(-1, -1, d)

        # pull top-k tokens from the original inputs
        filtered_inputs = torch.gather(
            input=inputs, dim=1, index=indices_expanded
        )  # -> batch, capacity, 1

        # slice an attention mask that matches the top-k selections
        squeezed_indices = token_indices.squeeze(-1)
        filtered_attention_mask = None
        if attention_mask is not None:
            filtered_attention_mask = torch.gather(
                input=attention_mask,
                dim=1,
                index=squeezed_indices,
            )

        filtered_block_ids = torch.gather(
            input=block_ids,
            dim=1,
            index=squeezed_indices,
        )  # [batch, k]

        # pass the selected tokens through a transformer block
        layer_outputs, layer_kv, state_update, aux_loss = layer(
            filtered_inputs,
            filtered_attention_mask,
            past_key_values,
            current_state,
            current_depth,
            filtered_block_ids,
            token_weights,
        )

        # reintegrate the processed tokens with our residual stream
        outputs = torch.scatter(
            input=inputs,
            dim=1,
            index=indices_expanded,
            src=layer_outputs,
        )

        return outputs, layer_kv, state_update, aux_loss + router_loss

    def aux_loss(self, router_logits: torch.Tensor, selected_indices: torch.Tensor):
        router_targets = torch.zeros_like(router_logits)
        router_targets.scatter_(1, selected_indices, 1.0)
        # page 7: the aux loss centers the sigmoid of the router’s outputs around 0.5
        return F.binary_cross_entropy_with_logits(
            router_logits.view(-1), router_targets.view(-1)
        )
