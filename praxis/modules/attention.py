import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# from praxis import PraxisConfig


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
        self.differential = config.differential
        self.max_seq_len = config.context_length
        self.hidden_size = config.num_dims
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Query and key projections for differential heads
        multiplier = 2 if self.differential else 1
        self.query = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim * multiplier,
            bias=False,
        )
        self.key = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim * multiplier,
            bias=False,
        )
        self.value = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )

        # Force exploration of attention subnetworks
        self.dropout = nn.Dropout(config.dropout)

        # Lambda vectors per differential head and per head
        if self.differential:
            self.lambda_init = 0.8  # A good default, per the paper
            self.lambdas = nn.ParameterDict(
                dict(
                    q1=nn.Parameter(torch.randn(self.head_dim)),
                    q2=nn.Parameter(torch.randn(self.head_dim)),
                    k1=nn.Parameter(torch.randn(self.head_dim)),
                    k2=nn.Parameter(torch.randn(self.head_dim)),
                )
            )
            self.norm = nn.GroupNorm(
                num_groups=self.num_heads, num_channels=self.num_heads * self.head_dim
            )

        self.output = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Pre-compute the ALiBi slopes
        slopes = 2 ** (-8 * torch.arange(1, self.num_heads + 1) / self.num_heads)
        self.register_buffer("slopes", slopes)
        self.register_buffer(
            "positions", torch.arange(self.max_seq_len, dtype=torch.float32)
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        token_indices: Optional[Tensor] = None,
    ):
        batch_size, query_seq_len, _ = query.shape
        _, key_seq_len, _ = key.shape

        # Compute queries, keys, and values
        multiplier = 2 if self.differential else 1
        q = (
            self.query(query)
            .view(batch_size, -1, self.num_heads, multiplier * self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(key)
            .view(batch_size, -1, self.num_heads, multiplier * self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        reciprocal = 1.0 / math.sqrt(self.head_dim)
        if self.differential:
            # Split queries and keys
            Q1, Q2 = q[..., : self.head_dim], q[..., self.head_dim :]
            K1, K2 = k[..., : self.head_dim], k[..., self.head_dim :]

            # Compute differntial attention scores
            scores = [
                torch.matmul(Q1, K1.transpose(-2, -1)) * reciprocal,
                torch.matmul(Q2, K2.transpose(-2, -1)) * reciprocal,
            ]
        else:
            # Compute attention scores
            scores = [torch.matmul(q, k.transpose(-2, -1)) * reciprocal]

        # Start with standard ALiBi positions
        if torch.is_tensor(token_indices):
            # Use provided token indices
            positions = self.positions[token_indices]
        else:
            # Use standard sequence positions
            positions = (
                self.positions[:query_seq_len]
                .unsqueeze(0)
                .expand(batch_size, query_seq_len)
            )

        # Compute initial position differences
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        biases = self.slopes.view(1, self.num_heads, 1, 1) * pos_diff.unsqueeze(1)

        # Handle memory tokens if present
        if key_seq_len != query_seq_len:
            memory_len = key_seq_len - query_seq_len

            # Create empty biases tensor of full size
            padded_biases = torch.zeros(
                batch_size,
                self.num_heads,
                query_seq_len,
                key_seq_len,
                device=query.device,
                dtype=query.dtype,
            )

            # Place the computed biases in the correct position after memory tokens
            padded_biases[..., :, memory_len:] = biases.expand(
                -1, -1, -1, query_seq_len
            )
            biases = padded_biases

        scores = [score - biases for score in scores]

        # Apply masks
        if self.causal:
            causal_mask = (
                torch.triu(
                    torch.full((query_seq_len, key_seq_len), -1e9, device=query.device),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = [score + causal_mask for score in scores]

        if len(attention_mask.shape) == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif len(attention_mask.shape) == 3:
            attention_mask = attention_mask.unsqueeze(1)

        attention_mask = (1.0 - attention_mask) * -1e9
        scores = [score + attention_mask for score in scores]

        # Compute attention weights
        weights = [self.dropout(F.softmax(score, dim=-1)) for score in scores]

        # Compute attention weights
        diff_weights = weights[0]
        if self.differential:
            # Compute scalar lambda
            lambda_scalar = (
                torch.exp(torch.dot(self.lambdas["q1"], self.lambdas["k1"]))
                - torch.exp(torch.dot(self.lambdas["q2"], self.lambdas["k2"]))
                + self.lambda_init
            )
            diff_weights = weights[0] - lambda_scalar * weights[1]

        # Compute attention output
        attention_scores = torch.matmul(
            diff_weights, v
        )  # Shape: (batch_size, num_heads, seq_len, head_dim)

        if self.differential:
            # Reshape for GroupNorm
            attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()
            # Shape: (batch_size, seq_len, num_heads, head_dim)

            attention_scores = attention_scores.view(
                batch_size, query_seq_len, self.num_heads * self.head_dim
            )
            # Shape: (batch_size, seq_len, num_heads * head_dim)

            # Permute to (batch_size, num_channels, seq_len)
            attention_scores = attention_scores.permute(0, 2, 1).contiguous()
            # Shape: (batch_size, num_heads * head_dim, seq_len)

            # Apply GroupNorm
            attention_scores = self.norm(attention_scores)

            # Permute back to (batch_size, seq_len, num_heads * head_dim)
            attention_scores = attention_scores.permute(0, 2, 1).contiguous()
            # Shape: (batch_size, seq_len, num_heads * head_dim)

            # Apply scaling factor
            attention_scores = attention_scores * (1 - self.lambda_init)
        else:
            attention_scores = attention_scores.transpose(1, 2).reshape(
                batch_size, -1, self.hidden_size
            )

        # Output projection
        return self.output(attention_scores)
