from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.activations import ACT2FN

from praxis.configuration_praxis import PraxisConfig


class PEER(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()

        n_dim = config.n_dim
        self.num_heads = config.peer_heads
        self.separate_embed_per_head = False
        self.num_experts = config.peer_experts
        self.num_experts_per_head = config.peer_experts_per_head
        product_key_topk = None
        dim_key = None

        num_expert_sets = self.num_heads if self.separate_embed_per_head else 1

        self.up_embed = nn.Embedding(self.num_experts * num_expert_sets, n_dim)
        self.down_embed = nn.Embedding(self.num_experts * num_expert_sets, n_dim)

        self.act = ACT2FN[config.activation]

        assert (
            self.num_experts**0.5
        ).is_integer(), "`self.num_experts` needs to be a square"
        assert (n_dim % 2) == 0, "Feature dimension should be divisible by 2"

        dim_key = self._default(dim_key, n_dim // 2)
        self.dim_key = dim_key  # Store as instance variable
        self.num_keys = int(self.num_experts**0.5)

        class Permute(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.permute(2, 0, 1, 3, 4).contiguous()

        self.to_queries = nn.Sequential(
            nn.Linear(n_dim, dim_key * self.num_heads * 2, bias=False),
            nn.Unflatten(-1, (2, self.num_heads, dim_key)),
            Permute(),
        )

        self.product_key_topk = self._default(
            product_key_topk, self.num_experts_per_head
        )

        scale = 0.02
        self.keys = nn.Parameter(
            torch.randn(self.num_heads, self.num_keys, 2, dim_key) * scale
        )

    def _default(self, val, d):
        return val if self._exists(val) else d

    def _exists(self, val):
        return val is not None

    def forward(self, x: Tensor):

        queries = self.to_queries(x)  # Shape: (2, batch_size, seq_len, heads, dim_key)

        # Compute similarities using Einstein summation
        sim = torch.einsum("p b n h d, h k p d -> p b n h k", queries, self.keys)

        # For each partition, get top-k indices and scores
        (scores_x, scores_y), (indices_x, indices_y) = [
            s.topk(self.product_key_topk, dim=-1) for s in sim
        ]

        # Compute Cartesian product of top-k indices and scores
        all_scores = scores_x.unsqueeze(-1) + scores_y.unsqueeze(-2)
        all_indices = indices_x.unsqueeze(-1) * self.num_keys + indices_y.unsqueeze(-2)

        # Flatten last two dimensions
        all_scores = all_scores.view(*all_scores.shape[:-2], -1)
        all_indices = all_indices.view(*all_indices.shape[:-2], -1)

        # Get top num_experts_per_head from the Cartesian product
        scores, pk_indices = all_scores.topk(self.num_experts_per_head, dim=-1)
        indices = all_indices.gather(-1, pk_indices)

        if self.separate_embed_per_head:
            head_expert_offsets = (
                torch.arange(self.num_heads, device=x.device) * self.num_experts
            )
            indices = indices + head_expert_offsets.view(1, 1, -1, 1)

        # Lookup expert weights using embeddings
        weights_down = self.down_embed(pk_indices)
        weights_up = self.up_embed(pk_indices)

        # Compute expert outputs
        x = torch.einsum("b n d, b n h k d -> b n h k", x, weights_down)

        # Activate the inputs
        x = self.act(x)

        # Apply softmax to scores
        x = F.softmax(scores, dim=-1) * x

        # Aggregate expert outputs
        x = torch.einsum("b n h k, b n h k d -> b n d", x, weights_up)

        return x
