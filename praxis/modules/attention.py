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
            self.hidden_size, self.num_heads * self.head_dim, bias=True
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

    def forward(self, inputs, attention_mask=None):
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

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim)
        )

        # Compute ALiBi bias
        alibi_bias = self.slopes.unsqueeze(1).unsqueeze(1) * self.positions[
            :seq_len
        ].unsqueeze(0).unsqueeze(0)

        alibi_bias = alibi_bias.expand(self.num_heads, seq_len, seq_len)

        # Subtract biases from the scores
        scores -= alibi_bias

        # Apply the causal mask
        if self.causal:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=inputs.device)
            ).view(1, 1, seq_len, seq_len)

            # Generate Gaussian random values
            mean = -1e9
            soft_mask = torch.normal(
                mean=mean,
                std=abs(mean) * min(self.foresight, 0.1),
                size=(batch_size, self.num_heads, seq_len, seq_len),
                device=inputs.device,
            )

            # Create an index tensor for scatter_add
            index = (
                torch.arange(seq_len, device=inputs.device)
                .view(1, 1, 1, seq_len)
                .expand_as(scores)
            )

            # Apply the random foresight values where the causal mask is 0
            scores = scores.scatter_add_(
                -1,
                index,
                soft_mask * (1 - causal_mask),  # Only add penalties where mask is 0
            )

        weights = F.softmax(scores, dim=-1)
        outputs = (
            torch.matmul(weights, v)
            .transpose(1, 2)
            .reshape(batch_size, seq_len, self.hidden_size)
        )

        return self.out(outputs)
