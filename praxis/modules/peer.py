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
        key_dim = config.expert["key_dim"]
        self.num_heads = config.expert["n_head"]
        self.offset_heads = config.expert["offset_heads"]
        self.n_experts = config.expert["n_experts"]
        self.num_keys = int(self.n_experts**0.5)
        self.topk = config.expert["topk"]

        num_expert_sets = self.num_heads if self.offset_heads else 1

        self.up_embed = nn.Embedding(self.n_experts * num_expert_sets, n_dim)
        self.down_embed = nn.Embedding(self.n_experts * num_expert_sets, n_dim)

        self.act = ACT2FN["gelu_new"]

        assert (
            self.n_experts**0.5
        ).is_integer(), "`self.num_experts` needs to be a square"
        assert (n_dim % 2) == 0, "`n_dim` should be divisible by 2"

        class Permute(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.permute(2, 0, 1, 3, 4).contiguous()

        # BatchNorm for combined partitions and heads
        class BatchNorm1d(nn.BatchNorm1d):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, x):
                b, s, d = x.size()
                x = x.view(b * s, d)
                x = super().forward(x)
                return x.view(b, s, d)

        self.queries = nn.Sequential(
            nn.Linear(n_dim, key_dim * self.num_heads * 2, bias=False),
            BatchNorm1d(key_dim * self.num_heads * 2),
            nn.Unflatten(-1, (2, self.num_heads, key_dim)),
            Permute(),
        )

        scale = 0.02
        self.keys = nn.Parameter(
            torch.randn(self.num_heads, self.num_keys, 2, key_dim) * scale
        )

    def forward(self, x: Tensor):
        # Generate queries
        queries = self.queries(x)  # Shape: (2, batch_size, seq_len, heads, dim_key)

        # Compute similarities using Einstein summation
        sim = torch.einsum("p b n h d, h k p d -> p b n h k", queries, self.keys)

        # For each partition, get top-k indices and scores
        (scores_x, indices_x), (scores_y, indices_y) = [
            s.topk(self.topk, dim=-1) for s in sim
        ]

        # Compute Cartesian product of top-k indices and scores
        all_scores = scores_x.unsqueeze(-1) + scores_y.unsqueeze(-2)
        all_indices = indices_x.unsqueeze(-1) * self.num_keys + indices_y.unsqueeze(-2)

        # Flatten last two dimensions
        all_scores = all_scores.view(*all_scores.shape[:-2], -1)
        all_indices = all_indices.view(*all_indices.shape[:-2], -1)

        # Get top num_experts_per_head from the Cartesian product
        scores, pk_indices = all_scores.topk(self.topk, dim=-1)
        indices = all_indices.gather(-1, pk_indices)

        if self.offset_heads:
            head_expert_offsets = (
                torch.arange(self.num_heads, device=x.device) * self.n_experts
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
