import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..configuration_praxis import PraxisConfig


# This uses expert-choice routing, which was greatly preferred by the
# original authors: https://arxiv.org/abs/2404.02258
# reference: https://huggingface.co/blog/joey00072/mixture-of-depth-is-vibe
class PraxisMixtureOfDepths(nn.Module):
    """
    Paper: https://arxiv.org/abs/2404.02258
    """

    def __init__(
        self,
        block: nn.Module = None,
        config: PraxisConfig = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.block = block
        self.capacity = config.capacity
        self.router = nn.Linear(config.n_dim, 1, bias=True)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        *args,
        **kwargs,
    ):

        b, s, d = x.shape

        # S = seq_len, C = capacity  , C = int(seq_length * capacity_factor)
        #  page 6 above eq 1 | ( C<S ) | here top_k = beta
        top_k = int(s * self.capacity)

        # eq1 page 6
        # scaler weights for each token
        router_logits = self.router(x)  # (x) batch,seq_len,dim -> r batch,seq_len,1

        #  𝑟𝑙> 𝑃𝛽 (R)  ... eqution 1
        token_weights, token_index = torch.topk(
            torch.sigmoid(router_logits), top_k, dim=1, sorted=False
        )

        # since its auto regressive model we need to keep casual nature of it
        # that why we need sort the tokens by idx before we pass it to attn
        sorted_indices, sorting_indices = torch.sort(token_index, dim=1)

        # select idx for copying for original tensor
        indices_expanded = sorted_indices.expand(-1, -1, d)

        # This are fillted topk tokens with capactiy C
        filtered_x = torch.gather(
            input=x, dim=1, index=indices_expanded
        )  # -> batch, capacity, dim

        # softmax router weights
        token_weights = F.softmax(token_weights, dim=1)

        # selecting router wight by idx
        router_weights = torch.gather(token_weights, dim=1, index=sorting_indices)

        # pass the selected tokens through the transformer block
        block_outputs = self.block(
            filtered_x, attention_mask=attention_mask, router_weights=router_weights
        )

        # integrate the selected and residual tokens
        hidden_states = torch.scatter(
            input=x, dim=1, index=indices_expanded, src=block_outputs["hidden_states"]
        )

        # compute aux loss, in order to maintain causality
        aux_loss = self.aux_loss(router_logits, sorted_indices)

        return dict(hidden_states=hidden_states, aux_loss=aux_loss)

    def aux_loss(self, router_logits: torch.Tensor, selected_tokens: torch.Tensor):
        # Page 7, Section 3.5 sampling
        router_targets = torch.zeros_like(router_logits)
        router_targets.scatter_(1, selected_tokens, 1.0)
        return F.binary_cross_entropy_with_logits(
            router_logits.view(-1), router_targets.view(-1)
        )
