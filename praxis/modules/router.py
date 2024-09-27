import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..configuration_praxis import PraxisConfig


# This uses expert-choice routing, which was greatly preferred by the
# original authors of this research: https://arxiv.org/abs/2404.02258
class PraxisMixtureOfDepths(nn.Linear):
    """
    Paper: https://arxiv.org/abs/2404.02258
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

        # S = seq_len, C = capacity  , C = int(seq_length * capacity_factor)
        #  page 6 above eq 1 | ( C<S ) | here top_k = beta
        top_k = int(s * self.capacity)

        # eq1 page 6
        # scaler weights for each token
        router_logits = F.linear(
            inputs, self.weight
        )  # (x) batch,seq_len,dim -> r batch,seq_len,1

        #  𝑟𝑙> 𝑃𝛽 (R)  ... equation 1
        token_weights, token_index = torch.topk(
            # page 7: [aux] loss centers the sigmoid of the router’s outputs around 0.5
            torch.sigmoid(router_logits),
            top_k,
            dim=1,
            sorted=False,
        )

        # since its auto regressive model we need to keep casual nature of it
        # that why we need sort the tokens by idx before we pass it to attn
        sorted_indices, sorting_indices = torch.sort(token_index, dim=1)

        # select idx for copying for original tensor
        indices_expanded = sorted_indices.expand(-1, -1, d)

        # This are fillted topk tokens with capactiy C
        filtered_inputs = torch.gather(
            input=inputs, dim=1, index=indices_expanded
        )  # -> batch, capacity, dim

        # softmax router weights
        token_weights = F.softmax(token_weights, dim=1)

        # selecting router wight by idx
        router_weights = torch.gather(token_weights, dim=1, index=sorting_indices)

        # pass the selected tokens through the transformer block
        layer_outputs = expert(
            filtered_inputs,
            attention_mask=attention_mask,
            router_weights=router_weights,
        )

        # integrate the selected and residual tokens
        outputs = torch.scatter(
            input=inputs,
            dim=1,
            index=indices_expanded,
            src=layer_outputs["hidden_states"],
        )

        # compute aux loss, in order to maintain causality
        aux_loss = self.aux_loss(router_logits, sorted_indices)

        return dict(hidden_states=outputs, aux_loss=aux_loss)

    def aux_loss(self, router_logits: torch.Tensor, selected_tokens: torch.Tensor):
        # Page 7, Section 3.5 sampling
        router_targets = torch.zeros_like(router_logits)
        router_targets.scatter_(1, selected_tokens, 1.0)
        return F.binary_cross_entropy_with_logits(
            router_logits.view(-1), router_targets.view(-1)
        )
