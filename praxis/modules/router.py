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

    def forward(
        self,
        inputs: Tensor,
        expert: nn.Module,
        attention_mask: Tensor,
        *args,
        **kwargs,
    ):

        b, s, d = inputs.shape

        # scalar weights for each token
        router_logits = F.linear(inputs, self.weight)  # -> batch, seq_len, 1

        k = int(s * self.capacity)

        if self.training:
            #  𝑟𝑙> 𝑃𝛽 (R) - equation 1
            token_weights, token_indices = torch.topk(
                router_logits,
                k,
                dim=1,
                sorted=False,
            )
        else:
            # top-k breaks causality; a sigmoid operation allows us to sample
            # autoregressively regardless, during inference
            token_mask = torch.sigmoid(router_logits) > 0.5
            token_indices = torch.nonzero(token_mask, as_tuple=True)[1].view(b, -1)

            if token_indices.numel() > 0:
                token_weights = (
                    router_logits.squeeze(-1).gather(1, token_indices).unsqueeze(-1)
                )
            else:
                # if no tokens were selected, just use the most recent k tokens
                selected_tokens = min(k, s)
                token_indices = (
                    torch.arange(s - selected_tokens, s, device=inputs.device)
                    .view(1, -1)
                    .expand(b, -1)
                )
                token_weights = torch.ones(b, selected_tokens, 1, device=inputs.device)

            token_indices = token_indices.unsqueeze(-1)

        # select idx for copying for original tensor
        indices_expanded = token_indices.expand(-1, -1, d)

        # filtered topk tokens with a capacity of C
        filtered_inputs = torch.gather(
            input=inputs, dim=1, index=indices_expanded
        )  # -> batch, capacity, 1

        # slice the attention mask based on the selected token indices
        filtered_attention_mask = torch.gather(
            input=attention_mask,
            dim=1,
            index=token_indices.squeeze(-1),
        )

        # pass the selected tokens through the transformer block
        expert_outputs = expert(
            filtered_inputs,
            attention_mask=filtered_attention_mask,
            router_weights=token_weights,
        )

        # integrate the activated tokens with the residual stream
        outputs = torch.scatter(
            input=inputs,
            dim=1,
            index=indices_expanded,
            src=expert_outputs["hidden_states"],
        )

        # compute aux loss, in order to maintain causality
        aux_loss = self.aux_loss(router_logits, token_indices)

        return dict(hidden_states=outputs, aux_loss=aux_loss)

    def aux_loss(self, router_logits: torch.Tensor, selected_indices: torch.Tensor):
        router_targets = torch.zeros_like(router_logits)
        router_targets.scatter_(1, selected_indices, 1.0)
        # page 7: the aux loss centers the sigmoid of the router’s outputs around 0.5
        return F.binary_cross_entropy_with_logits(
            router_logits.view(-1), router_targets.view(-1)
        )
