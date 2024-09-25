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
        self.register_buffer("m", self._get_alibi_slope(self.num_heads))

        # Precompute the relative positions for the maximum sequence length
        relative_positions = self._get_relative_positions(self.max_seq_len)

        # Compute the biases and store them
        self.register_buffer("alibi_bias", self.m * relative_positions)

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

        # Add biases to the scores
        scores += self.alibi_bias[:, :seq_len, :seq_len]

        # Apply the causal mask
        if self.causal:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device)
            ).view(1, 1, seq_len, seq_len)
            scores = scores.masked_fill(causal_mask == 0, self.foresight)

        # if attention_mask is not None:
        #     # Slice the attention mask to match the sequence length
        #     attention_mask = attention_mask[:, :seq_len]
        #     # Ensure attention_mask is broadcastable
        #     attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #     # attention_mask = (1.0 - attention_mask) * torch.finfo(scores.dtype).min
        #     scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.hidden_size
        )

        return self.out(attn_output)

    def _get_relative_positions(self, max_seq_len: int) -> torch.Tensor:
        # Compute the relative positions for the maximum sequence length
        # Shape: (1, max_seq_len, max_seq_len)
        relative_positions = torch.arange(max_seq_len, dtype=torch.long)
        return relative_positions[None, :] - relative_positions[:, None]

    def _get_alibi_slope(self, num_heads):
        start_slope = 2 ** (-8 / num_heads)
        slopes = torch.tensor([start_slope**i for i in range(1, num_heads + 1)]).float()
        return slopes.unsqueeze(1).unsqueeze(1)
