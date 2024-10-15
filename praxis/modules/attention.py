from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis import PraxisConfig


class PraxisAttention(nn.Module):
    """
    We implement Differential Attention, to filter the noise from attention maps:
    https://arxiv.org/abs/2410.05258

    We implement ALiBi for length extrapolation, to keep parameter counts low:
    https://arxiv.org/abs/2108.12409
    """

    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.causal = config.causal
        self.max_seq_len = config.context_length
        self.hidden_size = config.num_dims
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.effective_heads = config.differential_heads + 1

        # Query and key projections for differential heads
        self.query = nn.ModuleList(
            nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            for _ in range(self.effective_heads)
        )
        self.key = nn.ModuleList(
            nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            for _ in range(self.effective_heads)
        )
        self.value = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )

        # Force exploration of attention subnetworks
        self.dropout = nn.Dropout(config.dropout)

        # Lambda vectors per differential head and per head
        if self.effective_heads > 1:
            self.lambda_init = 0.8  # A good default, per the paper
            self.lambdas = nn.ParameterList(
                nn.Parameter(torch.randn(self.num_heads, self.head_dim))
                for _ in range(self.effective_heads * 2)
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
        self, inputs: Tensor, attention_mask: Tensor, token_indices: Optional[Tensor]
    ):
        batch_size, seq_len, _ = inputs.shape

        # Compute queries, keys, and values
        q = [
            self.query[i](inputs)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)  # Shape: (batch_size, num_heads, seq_len, head_dim)
            for i in range(self.effective_heads)
        ]
        k = [
            self.key[i](inputs)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            for i in range(self.effective_heads)
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
            for i in range(self.effective_heads)
        ]

        # Compute ALiBi biases
        if torch.is_tensor(token_indices):
            positions = self.positions[token_indices]
        else:
            positions = (
                self.positions[:seq_len].unsqueeze(0).expand(batch_size, seq_len)
            )

        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        biases = self.slopes.view(1, self.num_heads, 1, 1) * pos_diff.unsqueeze(1)
        scores = [scores[i] - biases for i in range(self.effective_heads)]

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
            scores = [scores[i] + causal_mask for i in range(self.effective_heads)]

        attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        scores = [scores[i] + attention_mask for i in range(self.effective_heads)]

        # Compute attention weights
        weights = [
            self.dropout(F.softmax(scores[i], dim=-1))
            for i in range(self.effective_heads)
        ]

        # return early if we aren't using differential attention
        if self.effective_heads == 1:
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
        for i in range(0, self.effective_heads, 2):
            # Compute dot product across head_dim for each head
            dot_product = (self.lambdas[i] * self.lambdas[i + 1]).sum(
                dim=-1
            )  # Shape: (num_heads,)

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
        for i in range(1, self.effective_heads):
            sign = (-1) ** i
            diff_weights += sign * lamb_expanded * weights[i]

        # Compute attention output
        attention_scores = torch.matmul(
            diff_weights, v
        )  # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Reshape for GroupNorm
        attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()
        # Shape: (batch_size, seq_len, num_heads, head_dim)

        attention_scores = attention_scores.view(
            batch_size, seq_len, self.num_heads * self.head_dim
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

        # Output projection
        return self.output(attention_scores)
