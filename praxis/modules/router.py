import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..configuration_praxis import PraxisConfig


class PraxisMixtureOfDepths(nn.Linear):
    """
    This uses expert-choice routing, which was greatly preferred by the
    original authors of this research: https://arxiv.org/abs/2404.02258
    """

    def __init__(
        self,
        config: PraxisConfig = None,
        *args,
        **kwargs,
    ):
        super().__init__(in_features=config.n_dim, out_features=1)
        self.capacity = config.capacity
        assert (
            self.capacity > 0 and self.capacity < 1.0
        ), "'capacity' must be set to a value between 0 and 1."

    def forward(
        self,
        expert: nn.Module,
        inputs: Tensor,
        attention_mask: Tensor,
        *args,
        **kwargs,
    ):

        b, s, d = inputs.shape
        k = int(s * self.capacity)

        # emit scalar weights for each token
        router_logits = F.linear(inputs, self.weight, self.bias)  # -> batch, seq_len, 1

        # the `b > 1` condition is required for sanity checking in Pytorch Lightning
        if self.training or b > 1:
            #  𝑟𝑙> 𝑃𝛽 (R) - equation 1
            token_weights, token_indices = torch.topk(
                router_logits,
                k,
                dim=1,
                sorted=False,
            )
        else:
            # top-k can see into the future, breaking causality; a sigmoid operation
            # allows us to sample autoregressively during inference
            token_mask = torch.sigmoid(router_logits) > 0.5
            token_indices = torch.nonzero(token_mask, as_tuple=True)[1].view(b, -1)

            if token_indices.numel() > 0:
                token_weights = (
                    router_logits.squeeze(-1).gather(1, token_indices).unsqueeze(-1)
                )
            else:
                # if no tokens were selected by the router, just use the most recent k tokens
                selected_tokens = min(k, s)
                token_indices = (
                    torch.arange(s - selected_tokens, s, device=inputs.device)
                    .view(1, -1)
                    .expand(b, -1)
                )
                token_weights = torch.ones(b, selected_tokens, 1, device=inputs.device)

            token_indices = token_indices.unsqueeze(-1)

        # expand router predictions to match input dimensions
        indices_expanded = token_indices.expand(-1, -1, d)

        # pull top-k tokens from the original inputs
        filtered_inputs = torch.gather(
            input=inputs, dim=1, index=indices_expanded
        )  # -> batch, capacity, 1

        # slice an attention mask that matches the top-k selections
        filtered_attention_mask = torch.gather(
            input=attention_mask,
            dim=1,
            index=token_indices.squeeze(-1),
        )

        # pass the selected tokens through a transformer block
        expert_outputs, _ = expert(
            filtered_inputs,
            attention_mask=filtered_attention_mask,
            router_weights=token_weights,
            token_indices=token_indices.squeeze(-1),
        )

        # re-integrate the activated tokens with our residual stream
        hidden_states = torch.scatter(
            input=inputs,
            dim=1,
            index=indices_expanded,
            src=expert_outputs,
        )

        # compute aux loss, in order to teach the router about causality
        aux_loss = self.aux_loss(router_logits, token_indices)

        return hidden_states, aux_loss

    def aux_loss(self, router_logits: torch.Tensor, selected_indices: torch.Tensor):
        router_targets = torch.zeros_like(router_logits)
        router_targets.scatter_(1, selected_indices, 1.0)
        # page 7: the aux loss centers the sigmoid of the router’s outputs around 0.5
        return F.binary_cross_entropy_with_logits(
            router_logits.view(-1), router_targets.view(-1)
        )
