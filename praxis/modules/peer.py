from typing import OrderedDict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.activations import ACT2FN

from praxis import PraxisConfig


class PraxisPEER(nn.Module):
    """
    This class implements the Parameter-Efficient Expert Retrieval (PEER) mechanism:
    https://arxiv.org/abs/2407.04153v1
    """

    def __init__(self, config: PraxisConfig):
        super().__init__()

        num_dims = config.num_dims
        key_dims = config.expert["key_dims"]
        self.num_heads = config.expert["num_heads"]
        self.offset_heads = config.expert["offset_heads"]
        self.num_experts = config.expert["num_experts"]
        self.num_keys = int(math.sqrt(self.num_experts))
        self.k = config.expert["k"]

        assert (
            self.num_experts**0.5
        ).is_integer(), "`self.num_experts` needs to be a square"
        assert (num_dims % 2) == 0, "`num_dims` should be divisible by 2"

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
                b, s, d = x.shape
                x = x.view(b * s, d)
                x = super().forward(x)
                return x.view(b, s, d)

        self.queries = nn.Sequential(
            BatchNorm1d(num_dims),
            nn.Linear(num_dims, key_dims * self.num_heads * 2, bias=False),
            # BatchNorm1d(key_dims * self.num_heads * 2),
            nn.Unflatten(-1, (2, self.num_heads, key_dims)),
            Permute(),
        )

        scale = 0.02
        self.keys = nn.Parameter(
            torch.randn(self.num_heads, self.num_keys, 2, key_dims) * scale
        )

        num_expert_sets = self.num_heads if self.offset_heads else 1
        self.key_in = nn.Embedding(self.num_experts * num_expert_sets, num_dims)
        self.act = ACT2FN[config.activation]
        self.key_out = nn.Embedding(self.num_experts * num_expert_sets, num_dims)

    def forward(self, inputs: Tensor):
        # Generate queries
        queries = self.queries(
            inputs
        )  # Shape: (2, batch_size, seq_len, heads, dim_key)

        # Compute similarities using Einstein summation
        sim = torch.einsum("p b n h d, h k p d -> p b n h k", queries, self.keys)

        # For each partition, get top-k indices and scores
        (scores_x, scores_y), (indices_x, indices_y) = sim.topk(self.k, dim=-1)

        # Compute Cartesian product of top-k indices and scores
        all_scores = scores_x.unsqueeze(-1) + scores_y.unsqueeze(-2)
        all_indices = indices_x.unsqueeze(-1) * self.num_keys + indices_y.unsqueeze(-2)

        # Flatten last two dimensions
        all_scores = all_scores.view(
            *all_scores.shape[:-2], math.prod(all_scores.shape[-2:])
        )
        all_indices = all_indices.view(
            *all_indices.shape[:-2], math.prod(all_indices.shape[-2:])
        )

        # Get top expert keys from the Cartesian product
        scores, pk_indices = all_scores.topk(self.k, dim=-1)
        indices = all_indices.gather(-1, pk_indices)

        if self.offset_heads:
            head_expert_offsets = (
                torch.arange(self.num_heads, device=inputs.device) * self.num_experts
            )
            indices = indices + head_expert_offsets.view(1, 1, -1, 1)

        # Lookup expert weights using embeddings
        weights_down = self.key_in(indices)
        weights_up = self.key_out(indices)

        # Compute expert outputs
        outputs = torch.einsum("b n d, b n h k d -> b n h k", inputs, weights_down)

        # Activate the inputs
        outputs = self.act(outputs)

        # Apply softmax to scores
        outputs = F.softmax(scores, dim=-1) * outputs

        # Aggregate expert outputs
        outputs = torch.einsum("b n h k, b n h k d -> b n d", outputs, weights_up)

        return outputs
