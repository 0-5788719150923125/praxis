from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.activations import ACT2FN

from praxis import PraxisConfig


class PEER(nn.Module):
    """
    This class implements the Parameter-Efficient Expert Retrieval (PEER) mechanism:
    https://arxiv.org/abs/2407.04153v1
    """

    def __init__(self, config: PraxisConfig):
        super().__init__()

        n_dim = config.n_dim
        key_dim = None
        product_key_topk = None
        self.num_heads = config.peer_heads
        self.separate_embed_per_head = True
        self.num_experts = config.peer_experts
        self.num_experts_per_head = config.peer_experts_per_head
        self.key_dim = key_dim if key_dim is not None else n_dim // 2
        self.product_key_topk = (
            product_key_topk
            if product_key_topk is not None
            else self.num_experts_per_head
        )

        num_expert_sets = self.num_heads if self.separate_embed_per_head else 1

        self.up_embed = nn.Embedding(self.num_experts * num_expert_sets, n_dim)
        self.down_embed = nn.Embedding(self.num_experts * num_expert_sets, n_dim)

        self.act = ACT2FN[config.activation]

        assert (
            self.num_experts**0.5
        ).is_integer(), "`self.num_experts` needs to be a square"
        assert (n_dim % 2) == 0, "`n_dim` should be divisible by 2"

        self.num_keys = int(self.num_experts**0.5)

        class Permute(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.permute(2, 0, 1, 3, 4).contiguous()

        # # BatchNorm for combined partitions and heads
        # self.norm = nn.BatchNorm1d(2 * self.num_heads * self.key_dim)

        self.queries = nn.Sequential(
            nn.Linear(n_dim, self.key_dim * self.num_heads * 2, bias=False),
            # nn.BatchNorm1d(2 * self.num_heads * self.key_dim),
            nn.Unflatten(-1, (2, self.num_heads, self.key_dim)),
            Permute(),
        )

        scale = 0.02
        self.keys = nn.Parameter(
            torch.randn(self.num_heads, self.num_keys, 2, self.key_dim) * scale
        )

    def forward(self, x: Tensor):

        batch_size, seq_len, _ = x.size()

        # Generate queries
        queries = self.queries(x)  # Shape: (2, batch_size, seq_len, heads, dim_key)

        # # Reshape for batch normalization
        # queries = queries.permute(
        #     0, 1, 3, 2, 4
        # )  # Shape: (batch_size, seq_len, num_heads, 2, dim_key)
        # queries = queries.contiguous().view(
        #     batch_size * seq_len, self.num_heads * 2 * self.key_dim
        # )

        # # Apply batch normalization
        # queries = self.norm(queries)

        # # Reshape back to original dimensions
        # queries = queries.view(batch_size, seq_len, self.num_heads, 2, self.key_dim)
        # queries = queries.permute(
        #     3, 0, 1, 2, 4
        # )  # Shape: (2, batch_size, seq_len, num_heads, dim_key)

        # Compute similarities using Einstein summation
        sim = torch.einsum("p b n h d, h k p d -> p b n h k", queries, self.keys)

        # For each partition, get top-k indices and scores
        (scores_x, indices_x), (scores_y, indices_y) = [
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
        weights_down = self.down_embed(indices)
        weights_up = self.up_embed(indices)

        # Compute expert outputs
        x = torch.einsum("b n d, b n h k d -> b n h k", x, weights_down)

        # Activate the inputs
        x = self.act(x)

        # Apply softmax to scores
        x = F.softmax(scores, dim=-1) * x

        # Aggregate expert outputs
        x = torch.einsum("b n h k, b n h k d -> b n d", x, weights_up)

        return x
