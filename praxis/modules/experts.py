from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.moe.server.layers.custom_experts import register_expert_class
from torch import Tensor
from transformers.activations import ACT2FN

from ..configuration_praxis import PraxisConfig
from .attention import PraxisAttention

input_shape = lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim))


@register_expert_class("praxis_block", input_shape)
class PraxisBlock(nn.Module):
    """
    A standard transformer block, which we typically refer to as an
    "expert" elsewhere.
    """

    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.n_dim, eps=config.epsilon)
        self.attn = PraxisAttention(config)
        self.mlp_norm = nn.RMSNorm(config.n_dim, eps=config.epsilon)
        self.mlp = PraxisGLU(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Tensor,
        router_weights: Tensor = None,
        token_indices: Tensor = None,
    ):
        residual = inputs
        normalized = self.attn_norm(inputs)
        outputs = self.attn(normalized, attention_mask, token_indices)
        outputs = outputs + residual
        residual = outputs
        normalized = self.mlp_norm(outputs)
        outputs = self.mlp(normalized)
        outputs = self.drop(outputs)
        if torch.is_tensor(router_weights):
            outputs *= router_weights
        aux_loss = 0
        outputs = outputs + residual
        return outputs, aux_loss


@register_expert_class("praxis_mlp", input_shape)
class PraxisMLP(nn.Sequential):
    def __init__(self, config: PraxisConfig):
        super().__init__(
            OrderedDict(
                [
                    ("up", nn.Linear(config.n_dim, 4 * config.n_dim)),
                    ("act", ACT2FN[config.activation]),
                    ("down", nn.Linear(4 * config.n_dim, config.n_dim)),
                ]
            )
        )


@register_expert_class("praxis_glu", input_shape)
class PraxisGLU(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.up = nn.Linear(config.n_dim, 8 * config.n_dim)
        self.act = ACT2FN[config.activation]
        self.down = nn.Linear(4 * config.n_dim, config.n_dim)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(a * self.act(b))


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class Permute(nn.Module):
    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1, 3, 4).contiguous()


class PEER(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads=8,
        num_experts=1_000_000,
        num_experts_per_head=16,
        activation=nn.GELU,
        dim_key=None,
        product_key_topk=None,
        separate_embed_per_head=False,
        pre_rmsnorm=False,
        dropout=0.0,
    ):
        super().__init__()

        self.norm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        self.heads = heads
        self.separate_embed_per_head = separate_embed_per_head
        self.num_experts = num_experts

        num_expert_sets = heads if separate_embed_per_head else 1

        self.weight_down_embed = nn.Embedding(num_experts * num_expert_sets, dim)
        self.weight_up_embed = nn.Embedding(num_experts * num_expert_sets, dim)

        self.activation = activation()

        assert (num_experts**0.5).is_integer(), "`num_experts` needs to be a square"
        assert (dim % 2) == 0, "Feature dimension should be divisible by 2"

        dim_key = default(dim_key, dim // 2)
        self.dim_key = dim_key  # Store as instance variable
        self.num_keys = int(num_experts**0.5)

        self.to_queries = nn.Sequential(
            nn.Linear(dim, dim_key * heads * 2, bias=False),
            nn.Unflatten(-1, (2, heads, dim_key)),
            Permute(),
            # Shape after permute: (p, b, n, h, d)
        )

        self.product_key_topk = default(product_key_topk, num_experts_per_head)
        self.num_experts_per_head = num_experts_per_head

        self.keys = nn.Parameter(torch.randn(heads, self.num_keys, 2, dim_key))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)

        queries = self.to_queries(x)  # Shape: (2, batch_size, seq_len, heads, dim_key)

        # Transpose keys to match dimensions
        keys = self.keys.permute(2, 0, 1, 3)  # Shape: (2, heads, num_keys, dim_key)

        # Compute similarities using torch.einsum
        sim = torch.einsum("p b n h d, p h k d -> p b n h k", queries, keys)

        # For each of the 2 partitions, get top-k indices and scores
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
                torch.arange(self.heads, device=x.device) * self.num_experts
            )
            indices = indices + head_expert_offsets.view(1, 1, -1, 1)

        # Lookup expert weights using embeddings
        indices_flat = indices.view(-1)
        weights_down = self.weight_down_embed(indices_flat)
        weights_up = self.weight_up_embed(indices_flat)

        # Reshape weights to match dimensions
        weights_down = weights_down.view(
            *indices.shape, -1
        )  # Shape: (batch_size, seq_len, heads, num_experts_per_head, dim)
        weights_up = weights_up.view(*indices.shape, -1)

        # Compute expert outputs
        x_expanded = x.unsqueeze(2).unsqueeze(
            3
        )  # Shape: (batch_size, seq_len, 1, 1, dim)
        x = torch.einsum(
            "b n d, b n h k d -> b n h k",
            x_expanded.squeeze(2).squeeze(2),
            weights_down,
        )

        x = self.activation(x)
        x = self.dropout(x)

        # Apply softmax to scores
        scores = F.softmax(scores, dim=-1)

        x = x * scores

        # Aggregate expert outputs
        x = torch.einsum("b n h k, b n h k d -> b n d", x, weights_up)

        return x


if __name__ == "__main__":
    # Parameters for the test
    dim = 64  # Model dimension, must be divisible by 2
    heads = 4
    num_experts = 256  # Must be a perfect square (e.g., 16^2)
    num_experts_per_head = 4
    seq_len = 10
    batch_size = 2

    # Create an instance of the PEER layer with reduced parameters
    peer_layer = PEER(
        dim=dim,
        heads=heads,
        num_experts=num_experts,
        num_experts_per_head=num_experts_per_head,
        activation=nn.GELU,
        dim_key=None,  # Defaults to dim // 2
        product_key_topk=None,  # Defaults to num_experts_per_head
        separate_embed_per_head=False,
        pre_rmsnorm=False,
        dropout=0.0,
    )

    # Generate dummy input data
    x = torch.randn(batch_size, seq_len, dim)

    # Run a forward pass
    output = peer_layer(x)

    # Print the output shape
    print("Output shape:", output.shape)  # Should be (batch_size, seq_len, dim)

    # Verify that the output shape matches the input shape
    assert output.shape == x.shape, "Output shape does not match input shape"
    print("Test passed: Output shape matches input shape.")
