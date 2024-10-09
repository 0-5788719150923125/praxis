import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_praxis import PraxisConfig


class PraxisAttention(nn.Module):
    """
    We implement Differential Attention, to filter the noise from attention maps:
    https://arxiv.org/abs/2410.05258

    We also implement ALiBi, to keep parameter counts low:
    https://arxiv.org/abs/2108.12409
    """

    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.causal = config.causal
        self.max_seq_len = config.context_length
        self.hidden_size = config.n_dim
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.differential_heads = config.differential_heads
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
        self.output = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Pre-compute the ALiBi slopes
        slopes = 2 ** (-8 * torch.arange(1, self.num_heads + 1) / self.num_heads)
        self.register_buffer("slopes", slopes)
        self.register_buffer(
            "positions", torch.arange(self.max_seq_len, dtype=torch.float32)
        )

        self.lambda_init = 0.8
        self.lambdas = [
            nn.Parameter(torch.randn(self.head_dim))
            for _ in range(self.differential_heads * 2)
        ]

        self.norm = nn.ModuleList(
            nn.LayerNorm(self.head_dim) for _ in range(self.num_heads)
        )

    def forward(self, inputs, attention_mask=None, token_indices=None):
        batch_size, seq_len, _ = inputs.size()

        q = [
            (
                self.query[i](inputs)
                .view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            for i in range(self.differential_heads)
        ]
        k = [
            (
                self.key[i](inputs)
                .view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            for i in range(self.differential_heads)
        ]
        v = (
            self.value(inputs)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = [
            torch.matmul(q[i], k[i].transpose(-2, -1))
            * torch.rsqrt(torch.tensor(self.head_dim, device=inputs.device))
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

        slopes = self.slopes.view(1, self.num_heads, 1, 1)
        biases = slopes * pos_diff.unsqueeze(1)

        scores = [scores[i] - biases for i in range(self.differential_heads)]

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
            scores = [scores[i] + causal_mask for i in range(self.differential_heads)]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = [scores[i] * mask for i in range(self.differential_heads)]

        # Compute softmax for all differential heads
        weights = [F.softmax(scores[i], dim=-1) for i in range(self.differential_heads)]

        # Set the initial weights
        diff_weights = weights[0]
        if len(weights) > 1:
            # Compute lambda
            lamb = 0
            for i in range(0, len(self.lambdas), 2):
                lamb = (
                    lamb
                    - torch.exp(torch.dot(self.lambdas[i], self.lambdas[i + 1]))
                    + self.lambda_init
                )
            # Compute the differential attention weights
            for i, w in enumerate(weights):
                if i + 1 >= len(weights):
                    break
                diff_weights = diff_weights - lamb * weights[i + 1]

        # Apply LayerNorm to each head
        attention_heads = torch.matmul(diff_weights, v).transpose(1, 2)
        attention_heads = torch.stack(
            [self.norm[i](attention_heads[..., i, :]) for i in range(self.num_heads)],
            dim=-2,
        )

        attention = attention_heads.reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )

        return self.output(attention)


# import math
# from typing import Optional, Tuple, Union

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from ..configuration_praxis import PraxisConfig


# class PraxisAttention(nn.Module):
#     def __init__(self, config: PraxisConfig):
#         super().__init__()
#         self.causal = config.causal
#         self.max_seq_len = config.context_length
#         self.hidden_size = config.n_dim
#         self.num_heads = config.n_head
#         self.head_dim = self.hidden_size // self.num_heads
#         self.query = nn.Linear(
#             self.hidden_size, self.num_heads * self.head_dim, bias=False
#         )
#         self.key = nn.Linear(
#             self.hidden_size, self.num_heads * self.head_dim, bias=False
#         )
#         self.value = nn.Linear(
#             self.hidden_size, self.num_heads * self.head_dim, bias=False
#         )
#         self.output = nn.Linear(
#             self.num_heads * self.head_dim, self.hidden_size, bias=False
#         )

#         # Pre-compute the ALiBi slopes
#         slopes = 2 ** (-8 * torch.arange(1, self.num_heads + 1) / self.num_heads)
#         self.register_buffer("slopes", slopes)
#         self.register_buffer(
#             "positions", torch.arange(self.max_seq_len, dtype=torch.float32)
#         )

#     def forward(self, inputs, attention_mask=None, token_indices=None):

#         batch_size, seq_len, _ = inputs.size()

#         q = (
#             self.query(inputs)
#             .view(batch_size, seq_len, self.num_heads, self.head_dim)
#             .transpose(1, 2)
#         )
#         k = (
#             self.key(inputs)
#             .view(batch_size, seq_len, self.num_heads, self.head_dim)
#             .transpose(1, 2)
#         )
#         v = (
#             self.value(inputs)
#             .view(batch_size, seq_len, self.num_heads, self.head_dim)
#             .transpose(1, 2)
#         )

#         scores = torch.matmul(q, k.transpose(-2, -1)) * torch.rsqrt(
#             torch.tensor(self.head_dim, device=inputs.device)
#         )

#         # Compute ALiBi biases
#         if token_indices is not None:
#             positions = self.positions[token_indices]  # [batch_size, seq_len]
#         else:
#             positions = (
#                 self.positions[:seq_len].unsqueeze(0).expand(batch_size, seq_len)
#             )  # [batch_size, seq_len]

#         # Compute position differences
#         pos_diff = positions.unsqueeze(2) - positions.unsqueeze(
#             1
#         )  # [batch_size, seq_len, seq_len]

#         # Compute biases
#         slopes = self.slopes.view(1, self.num_heads, 1, 1)  # [1, num_heads, 1, 1]
#         biases = slopes * pos_diff.unsqueeze(
#             1
#         )  # [batch_size, num_heads, seq_len, seq_len]

#         # Subtract biases from the scores
#         scores -= biases

#         # Apply the causal mask
#         if self.causal:
#             causal_mask = (
#                 torch.triu(
#                     torch.ones(seq_len, seq_len, device=inputs.device) * float("-inf"),
#                     diagonal=1,
#                 )
#                 .unsqueeze(0)
#                 .unsqueeze(0)
#             )
#             scores += causal_mask

#         if attention_mask is not None:
#             scores *= attention_mask.unsqueeze(1).unsqueeze(1)

#         weights = F.softmax(scores, dim=-1)
#         attention = (
#             torch.matmul(weights, v)
#             .transpose(1, 2)
#             .reshape(batch_size, seq_len, self.hidden_size)
#         )

#         return self.output(attention)
