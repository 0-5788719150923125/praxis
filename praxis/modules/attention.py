import math
from typing import Optional, OrderedDict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_praxis import PraxisConfig


class PraxisAttention(nn.Module):
    """
    We implement Differential Attention, to filter the noise from attention maps:
    https://arxiv.org/abs/2410.05258

    We implement ALiBi for length extrapolation, to keep parameter counts low:
    https://arxiv.org/abs/2108.12409
    """

    def __init__(self, config):
        super().__init__()
        self.causal = config.causal
        self.max_seq_len = config.context_length
        self.hidden_size = config.n_dim
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.differential_heads = config.differential_heads
        assert (
            self.differential_heads > 0
        ), "'differential_heads' must be set to a value greater than 0."

        # Query and key projections for differential heads
        self.query = nn.ModuleList(
            nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            for _ in range(self.differential_heads)
        )
        self.key = nn.ModuleList(
            nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            for _ in range(self.differential_heads)
        )
        self.value = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )

        # Lambda vectors per differential head and per head
        if self.differential_heads > 1:
            self.lambda_init = 0.8  # As per the paper
            self.lambdas = nn.ModuleDict(
                OrderedDict(
                    q=nn.ParameterList(
                        nn.Parameter(torch.randn(self.num_heads, self.head_dim))
                        for _ in range(self.differential_heads)
                    ),
                    k=nn.ParameterList(
                        nn.Parameter(torch.randn(self.num_heads, self.head_dim))
                        for _ in range(self.differential_heads)
                    ),
                )
            )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.head_dim)

        self.output = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Pre-compute the ALiBi slopes
        slopes = 2 ** (-8 * torch.arange(1, self.num_heads + 1) / self.num_heads)
        self.register_buffer("slopes", slopes)
        self.register_buffer(
            "positions", torch.arange(self.max_seq_len, dtype=torch.float32)
        )

    def forward(self, inputs, attention_mask=None, token_indices=None):
        batch_size, seq_len, _ = inputs.size()

        # Compute queries, keys, and values
        q = [
            self.query[i](inputs)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)  # Shape: (batch_size, num_heads, seq_len, head_dim)
            for i in range(self.differential_heads)
        ]
        k = [
            self.key[i](inputs)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            for i in range(self.differential_heads)
        ]
        v = (
            self.value(inputs)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores
        reciprocal = 1.0 / math.sqrt(self.head_dim)
        scores = [
            torch.matmul(q[i], k[i].transpose(-2, -1)) * reciprocal
            for i in range(self.differential_heads)
        ]

        # Compute ALiBi biases
        if token_indices is not None:
            positions = self.positions[token_indices]
        else:
            positions = (
                self.positions[:seq_len].unsqueeze(0).expand(batch_size, seq_len)
            )

        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        biases = self.slopes.view(1, self.num_heads, 1, 1) * pos_diff.unsqueeze(1)
        scores = [scores[i] - biases for i in range(self.differential_heads)]

        # Apply masks
        if self.causal:
            causal_mask = (
                torch.triu(
                    torch.full((seq_len, seq_len), -1e9, device=inputs.device),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = [scores[i] + causal_mask for i in range(self.differential_heads)]

        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            scores = [
                scores[i] + attention_mask for i in range(self.differential_heads)
            ]

        # Compute attention weights
        weights = [F.softmax(scores[i], dim=-1) for i in range(self.differential_heads)]

        # return early if we aren't using differential attention
        if self.differential_heads == 1:
            # Use standard attention
            attention = (
                torch.matmul(weights[0], v)
                .transpose(1, 2)
                .reshape(batch_size, seq_len, self.hidden_size)
            )
            # Output projection
            return self.output(attention)

        # Compute scalar lambdas per head via dot products
        lambda_scalars = []
        for i in range(self.differential_heads):
            # Shape: (num_heads, head_dim)
            lambda_q_i = self.lambdas["q"][i]
            lambda_k_i = self.lambdas["k"][i]

            # Compute dot product across head_dim for each head
            dot_product = (lambda_q_i * lambda_k_i).sum(dim=-1)  # Shape: (num_heads,)

            # Compute scalar lambda per head
            lambda_scalar = torch.exp(dot_product)  # Shape: (num_heads,)
            lambda_scalars.append(lambda_scalar)

        # Compute the overall lambda per head
        lamb = self.lambda_init * torch.ones(
            self.num_heads, device=inputs.device
        )  # Shape: (num_heads,)
        for i, lambda_scalar in enumerate(lambda_scalars):
            sign = (-1) ** i
            lamb += sign * lambda_scalar  # Shape: (num_heads,)

        # Expand lamb the lambda for broadcasting
        lamb_expanded = (
            lamb.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )  # (1, num_heads, 1, 1)

        # Compute differential attention weights
        diff_weights = lamb_expanded * weights[0]
        for i in range(1, self.differential_heads):
            sign = (-1) ** i
            diff_weights += sign * lamb_expanded * weights[i]

        # Compute attention output
        attention_scores = torch.matmul(
            diff_weights, v
        )  # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Reshape for GroupNorm
        attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()
        # Shape: (batch_size, seq_len, num_heads, head_dim)

        # Merge batch_size and seq_len
        attention_scores = attention_scores.view(-1, self.head_dim)
        # Shape: (batch_size * seq_len * num_heads, head_dim)

        # Apply GroupNorm
        attention_scores = self.norm(attention_scores)

        # Reshape back to (batch_size, seq_len, num_heads, head_dim)
        attention_scores = attention_scores.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        # Apply scaling factor
        attention_scores = attention_scores * (1 - self.lambda_init)

        # Reshape to (batch_size, seq_len, num_heads * head_dim)
        attention_scores = attention_scores.view(
            batch_size, seq_len, self.num_heads * self.head_dim
        )

        # Output projection
        return self.output(attention_scores)
