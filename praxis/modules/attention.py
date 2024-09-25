import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_praxis import PraxisConfig


class PraxisAttention(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.causal = config.causal
        self.max_seq_len = config.context_length
        self.foresight = config.foresight
        self.hidden_size = config.n_dim
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.query = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.key = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.value = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.out = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=True
        )

        # Precompute the slopes for ALiBi
        slopes = 2 ** (-8 * torch.arange(1, self.num_heads + 1) / self.num_heads)
        self.register_buffer("slopes", slopes)

        # Precompute the positions
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        self.register_buffer("positions", positions)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        q = (
            self.query(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim)
        )

        # Compute ALiBi bias
        alibi_bias = self.slopes.unsqueeze(1).unsqueeze(1) * self.positions[
            :seq_len
        ].unsqueeze(0).unsqueeze(0)
        alibi_bias = alibi_bias.expand(self.num_heads, seq_len, seq_len)

        # Subtract biases from the scores
        scores = scores - alibi_bias

        # Apply the causal mask
        if self.causal:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device)
            ).view(1, 1, seq_len, seq_len)
            scores = scores.masked_fill(causal_mask == 0, self.foresight)

        # Apply the causal mask
        # if self.causal:
        #     # Create the causal mask
        #     causal_mask = torch.triu(
        #         torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        #     ).bool()
        #     causal_mask = causal_mask.unsqueeze(0).unsqueeze(
        #         0
        #     )  # Add batch and head dimensions

        #     # Apply the penalty to the masked positions
        #     scores = torch.where(causal_mask, scores + self.foresight, scores)
        #     print(scores)

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = (
            torch.matmul(attn_weights, v)
            .transpose(1, 2)
            .reshape(batch_size, seq_len, self.hidden_size)
        )

        return self.out(attn_output)

    def _get_relative_positions(self, max_seq_len: int) -> torch.Tensor:
        # Compute the relative positions for the maximum sequence length
        relative_positions = torch.arange(max_seq_len, dtype=torch.long)
        return relative_positions[None, :] - relative_positions[:, None]

    def _get_alibi_slope(self, num_heads):
        start_slope = 2 ** (-8 / num_heads)
        slopes = 2 ** (-8 * torch.arange(1, num_heads + 1) / num_heads)
        return slopes
