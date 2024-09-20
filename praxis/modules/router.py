import torch
import torch.nn as nn
import torch.nn.functional as F


# This uses expert-choice routing, which was greatly preferred by the
# original authors: https://arxiv.org/abs/2404.02258
class PraxisMixtureOfDepths(nn.Module):
    def __init__(
        self,
        block: nn.Module = None,
        use_router: bool = False,
        n_dim: int = None,
        capacity_factor: float = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.block = block  # Transformer block to process selected tokens
        self.use_router = use_router
        self.n_dim = n_dim
        self.capacity_factor = capacity_factor  # Should be a float between 0 and 1

        if self.use_router:
            self.router = nn.Linear(n_dim, 1, bias=False)
            # Auxiliary predictor for autoregressive sampling (if needed)
            self.aux_router = nn.Sequential(
                nn.Linear(n_dim, n_dim // 2),
                nn.SiLU(),
                nn.Linear(n_dim // 2, 1),
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        freqs_cis: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        if not self.use_router:
            x = self.block(x)
            aux_loss = 0
            return x, aux_loss

        b, s, d = x.shape  # Batch size, sequence length, hidden dimension

        # Compute capacity (number of tokens to process per batch element)
        top_k = max(
            1, int(s * self.capacity_factor)
        )  # Ensure at least one token is selected

        # Compute router logits (scalar per token)
        router_logits = self.router(x).squeeze(-1)  # Shape: [b, s]

        # Determine top-k tokens based on router logits (expert-choice routing)
        _, selected_indices = torch.topk(
            router_logits, top_k, dim=-1
        )  # Shape: [b, top_k]

        # Create batch indices for indexing
        batch_indices = (
            torch.arange(b, device=x.device).unsqueeze(1).expand(-1, top_k)
        )  # Shape: [b, top_k]

        # Extract selected tokens
        x_selected = x[batch_indices, selected_indices]  # Shape: [b, top_k, d]

        # Multiply the processed tokens by the router weights (include in gradient path)
        router_weights = router_logits[batch_indices, selected_indices].unsqueeze(
            -1
        )  # Shape: [b, top_k, 1]

        # Process selected tokens
        x_processed = self.block(
            x_selected, weights=router_weights
        )  # Processed tokens, Shape: [b, top_k, d]

        # Prepare the output tensor
        out = x.clone()  # Start with residual tokens

        # Assign processed tokens back into their original positions
        out[batch_indices, selected_indices] = x_processed  # Shape matching is ensured

        # Auxiliary loss for router (if implementing predictive routing for sampling)
        aux_loss = self.aux_loss(x, router_logits, selected_indices, batch_indices)

        return out, aux_loss

    def aux_loss(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        selected_indices: torch.Tensor,
        batch_indices: torch.Tensor,
    ):
        b, s, d = x.shape

        # Create targets: 1 for selected tokens, 0 for others
        targets = torch.zeros_like(router_logits)  # Shape: [b, s]
        targets[batch_indices, selected_indices] = 1.0  # Mark selected tokens

        # Flatten tensors
        router_logits_flat = router_logits.view(-1)  # Shape: [b * s]
        targets_flat = targets.view(-1)  # Shape: [b * s]

        # Compute auxiliary router logits
        aux_router_logits = self.aux_router(x.detach().view(-1, self.n_dim)).squeeze(
            -1
        )  # Shape: [b * s]

        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(aux_router_logits, targets_flat)
        return loss


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# This is a token-choice router
# class PraxisMixtureOfDepths(nn.Module):
#     def __init__(
#         self,
#         block: nn.Module = None,
#         dim: int = None,
#         capacity_factor: float = None,
#         *args,
#         **kwargs,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.capacity_factor = capacity_factor  # Should be a float between 0 and 1
#         self.block = block
#         self.router = nn.Linear(dim, 1, bias=False)

#         self.aux_router = nn.Sequential(
#             nn.Linear(dim, dim // 2),
#             nn.SiLU(),
#             nn.Linear(dim // 2, 1),
#         )

#     def forward(
#         self,
#         x: torch.Tensor = None,
#         mask: torch.Tensor = None,
#         freqs_cis: torch.Tensor = None,
#         *args,
#         **kwargs,
#     ) -> torch.Tensor:

#         b, s, d = x.shape

#         # Top k
#         top_k = max(
#             1, int(s * self.capacity_factor)
#         )  # Ensure at least one token is selected

#         # Scalar weights for each token
#         router_logits = self.router(x).squeeze(-1)  # Shape: [b, s]

#         # Apply softmax to router logits
#         router_probs = F.softmax(router_logits, dim=-1)  # Shape: [b, s]

#         # Token-choice routing: each token chooses a path based on probability
#         # For training, you might sample from the distribution
#         # For deterministic behavior, select top_k tokens per batch element

#         # Get top_k tokens per batch
#         _, selected_indices = torch.topk(router_probs, top_k, dim=-1)

#         # Create a mask for selected tokens
#         mask = torch.zeros_like(router_probs, dtype=torch.bool)  # Shape: [b, s]
#         mask.scatter_(dim=-1, index=selected_indices, value=True)

#         # Process selected tokens
#         x_selected = x[mask].view(b, top_k, d)  # Shape: [b, top_k, d]
#         x_processed = self.block(x_selected)  # Processed tokens

#         # Prepare the output tensor
#         out = x.clone()  # Start with residual tokens

#         # Place the processed tokens back into their original positions
#         out[mask] = x_processed.view(-1, d)

#         # Auxiliary loss for router
#         aux_loss = self.aux_loss(x, router_logits, mask)

#         return out, aux_loss

#     def aux_loss(
#         self,
#         x: torch.Tensor,
#         router_logits: torch.Tensor,
#         mask: torch.Tensor,
#     ):
#         # Flatten tensors for loss computation
#         router_logits = router_logits.view(-1)  # Shape: [b*s]
#         mask = mask.view(-1).float()  # Convert mask to float tensor [0., 1.]

#         # Compute auxiliary router logits
#         aux_router_logits = self.aux_router(x.detach().view(-1, self.dim)).squeeze(
#             -1
#         )  # Shape: [b*s]

#         # Binary cross-entropy loss
#         loss = F.binary_cross_entropy_with_logits(aux_router_logits, mask)
#         return loss
