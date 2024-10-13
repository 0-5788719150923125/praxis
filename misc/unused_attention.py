import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_praxis import PraxisConfig


class PraxisAttention(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.causal = config.causal
        self.max_seq_len = config.context_length
        self.hidden_size = config.num_dims
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.query = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.key = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.value = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
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

    def forward(self, inputs, attention_mask=None, token_indices=None):

        batch_size, seq_len, _ = inputs.size()

        q = (
            self.query(inputs)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(inputs)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(inputs)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) * torch.rsqrt(
            torch.tensor(self.head_dim, device=inputs.device)
        )

        # Compute ALiBi biases
        if token_indices is not None:
            positions = self.positions[token_indices]  # [batch_size, seq_len]
        else:
            positions = (
                self.positions[:seq_len].unsqueeze(0).expand(batch_size, seq_len)
            )  # [batch_size, seq_len]

        # Compute position differences
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(
            1
        )  # [batch_size, seq_len, seq_len]

        # Compute biases
        slopes = self.slopes.view(1, self.num_heads, 1, 1)  # [1, num_heads, 1, 1]
        biases = slopes * pos_diff.unsqueeze(
            1
        )  # [batch_size, num_heads, seq_len, seq_len]

        # Subtract biases from the scores
        scores -= biases

        # Apply the causal mask
        if self.causal:
            causal_mask = (
                torch.triu(
                    torch.ones(seq_len, seq_len, device=inputs.device) * float("-inf"),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores += causal_mask

        if attention_mask is not None:
            scores *= attention_mask.unsqueeze(1).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        attention = (
            torch.matmul(weights, v)
            .transpose(1, 2)
            .reshape(batch_size, seq_len, self.hidden_size)
        )

        return self.output(attention)
