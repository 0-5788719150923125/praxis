import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig


class PraxisMixtureOfDepths(nn.Linear):
    """
    This uses expert-choice routing, which was greatly preferred by the
    original authors of this research: https://arxiv.org/abs/2404.02258
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__(in_features=config.hidden_size, out_features=1)
        self.config = config
        self.capacity = config.capacity
        assert (
            self.capacity > 0 and self.capacity < 1.0
        ), "'capacity' must be set to a value between 0 and 1."

    def forward(self, layer: nn.Module, inputs: Tensor, attention_mask: Tensor):

        b, s, d = inputs.shape
        k = int(s * self.capacity)

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
        aux_loss = self.aux_loss(router_logits, token_indices)

        # when inputs have a length of 1, the router will sometimes select no tokens at all
        if token_weights.size(1) == 0:
            return inputs, aux_loss

        # expand router predictions to match input dimensions
        indices_expanded = token_indices.expand(-1, -1, d)

        # pull top-k tokens from the original inputs
        filtered_inputs = torch.gather(
            input=inputs, dim=1, index=indices_expanded
        )  # -> batch, capacity, 1

        # slice an attention mask that matches the top-k selections
        squeezed_indices = token_indices.squeeze(-1)
        filtered_attention_mask = torch.gather(
            input=attention_mask,
            dim=1,
            index=squeezed_indices,
        )

        # pass the selected tokens through a transformer block
        layer_outputs = layer(filtered_inputs, filtered_attention_mask, token_weights)

        # reintegrate the processed tokens with our residual stream
        outputs = torch.scatter(
            input=inputs,
            dim=1,
            index=indices_expanded,
            src=layer_outputs,
        )

        return outputs, aux_loss

    def aux_loss(self, router_logits: torch.Tensor, selected_indices: torch.Tensor):
        router_targets = torch.zeros_like(router_logits)
        router_targets.scatter_(1, selected_indices, 1.0)
        # page 7: the aux loss centers the sigmoid of the router’s outputs around 0.5
        return F.binary_cross_entropy_with_logits(
            router_logits.view(-1), router_targets.view(-1)
        )
