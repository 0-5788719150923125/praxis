import torch.nn as nn


class PraxisMixtureOfDepths(nn.Module):
    def __init__(
        self,
        seq_len: int = None,
        dim: int = None,
        capacity_factor: int = None,
        block: nn.Module = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.capacity_factor = capacity_factor
        self.block = block
        self.router = nn.Linear(dim, 1, bias=False)

        self.aux_router = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(
        self,
        x: Tensor = None,
        mask: Tensor = None,
        freqs_cis: Tensor = None,
        *args,
        **kwargs,
    ) -> Tensor:

        b, s, d = x.shape
        device = x.device

        # Top k
        top_k = int(s * self.capacity_factor)

        # Scalar weights for each token
        router_logits = self.router(x)

        # Equation 1
        token_weights, token_index = torch.topk(
            router_logits, top_k, dim=1, sorted=False
        )

        # Selected
        selected_tokens, index = torch.sort(token_index, dim=1)

        # Select idx
        indices_expanded = selected_tokens.expand(-1, -1, self.dim)

        # Filtered topk tokens with capacity c
        filtered_x = torch.gather(input=x, dim=1, index=indices_expanded)
        # print(filtered_x.shape)

        # I think filtered_x goes through the transformer block?
        x_out = self.block(filtered_x)

        # Softmax router weights
        token_weights = F.softmax(token_weights, dim=1)

        # Selecting router weight by idx
        r_weights = torch.gather(token_weights, dim=1, index=index)

        # Multiply by router weights
        xw_out = r_weights * x_out

        # Out
        out = torch.scatter_add(input=x, dim=1, index=indices_expanded, src=xw_out)

        # Aux loss
        aux_loss = self.aux_loss(out, router_logits, selected_tokens)
        return out, aux_loss

    def aux_loss(
        self,
        x: Tensor,
        router_logits: Tensor,
        selected_tokens: Tensor,
    ):
        b, s, d = x.shape

        router_targets = torch.zeros_like(router_logits).view(-1)

        router_targets[selected_tokens.view(-1)] = 1.0
        aux_router_logits = self.aux_router(x.detach().view(b * s, -1))
        return F.binary_cross_entropy_with_logits(
            aux_router_logits.view(-1), router_targets
        )
