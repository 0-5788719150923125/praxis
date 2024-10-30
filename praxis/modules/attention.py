import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig


class PraxisAttention(nn.Module):
    """
    We implement Differential Attention, to filter the noise from attention maps:
    https://arxiv.org/abs/2410.05258

    We implement ALiBi for length extrapolation, to keep parameter counts low:
    https://arxiv.org/abs/2108.12409
    """

    def __init__(self, config: AutoConfig):
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

        # Add memory-related parameters
        self.use_memory = config.memory
        if self.use_memory:
            self.memory = PraxisMemory(config)

        # Standard output projection
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
        multiplier = 2 if self.differential else 1
        q = (
            self.query(inputs)
            .view(batch_size, -1, self.num_heads, multiplier * self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(inputs)
            .view(batch_size, -1, self.num_heads, multiplier * self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(inputs)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        reciprocal = 1.0 / math.sqrt(self.head_dim)
        if self.differential:
            # Split queries and keys
            Q1, Q2 = q[..., : self.head_dim], q[..., self.head_dim :]
            K1, K2 = k[..., : self.head_dim], k[..., self.head_dim :]

            # Compute differential attention scores
            scores = [
                torch.matmul(Q1, K1.transpose(-2, -1)) * reciprocal,
                torch.matmul(Q2, K2.transpose(-2, -1)) * reciprocal,
            ]
        else:
            # Compute attention scores
            scores = [torch.matmul(q, k.transpose(-2, -1)) * reciprocal]

        # Compute ALiBi biases
        if torch.is_tensor(token_indices):
            positions = self.positions[token_indices]
        else:
            positions = (
                self.positions[:seq_len].unsqueeze(0).expand(batch_size, seq_len)
            )

        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        biases = self.slopes.view(1, self.num_heads, 1, 1) * pos_diff.unsqueeze(1)
        scores = [score - biases for score in scores]

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
            scores = [score + causal_mask for score in scores]

        attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        scores = [score + attention_mask for score in scores]

        # Compute attention weights
        weights = [self.dropout(F.softmax(score, dim=-1)) for score in scores]

        # Compute attention weights
        attention_weights = weights[0]
        if self.differential:
            # Compute scalar lambda
            lambda_scalar = (
                torch.exp(torch.dot(self.lambdas["q1"], self.lambdas["k1"]))
                - torch.exp(torch.dot(self.lambdas["q2"], self.lambdas["k2"]))
                + self.lambda_init
            )
            attention_weights = weights[0] - lambda_scalar * weights[1]

        # Compute attention output
        attention_output = (
            attention_weights @ v
        )  # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Use differential attention
        if self.differential:
            # Apply GroupNorm to attention scores before memory computation
            # Reshape for GroupNorm
            attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
            # Shape: (batch_size, seq_len, num_heads, head_dim)
            attention_output = attention_output.view(
                batch_size, seq_len, self.num_heads * self.head_dim
            )
            # Shape: (batch_size, seq_len, num_heads * head_dim)
            # Permute to (batch_size, num_channels, seq_len)
            attention_output = attention_output.permute(0, 2, 1).contiguous()
            # Shape: (batch_size, num_heads * head_dim, seq_len)
            # Apply GroupNorm
            attention_output = self.norm(attention_output)
            # Permute back to (batch_size, seq_len, num_heads * head_dim)
            attention_output = attention_output.permute(0, 2, 1).contiguous()
            # Shape: (batch_size, seq_len, num_heads * head_dim)
            # Apply scaling factor
            attention_output = attention_output * (1 - self.lambda_init)
            # Reshape back to (batch_size, num_heads, seq_len, head_dim)
            attention_output = attention_output.view(
                batch_size, seq_len, self.num_heads, self.head_dim
            )
            attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
            # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Add memory-based attention
        if self.use_memory:
            attention_output = self.memory(q, k, v, attention_output)

        if not self.differential:
            attention_output = attention_output.transpose(1, 2).reshape(
                batch_size, seq_len, self.hidden_size
            )
        else:
            # Reshape to (batch_size, seq_len, num_heads * head_dim)
            attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
            attention_output = attention_output.view(
                batch_size, seq_len, self.num_heads * self.head_dim
            )

        # Output projection
        return self.output(attention_output)


class PraxisMemory(nn.Module):
    """
    We also implement a simplified version of Infini-Attention, which omits the chunking:
    https://arxiv.org/abs/2404.07143
    """

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.epsilon = 1e-8
        self.use_delta = True
        self.num_heads = config.num_heads
        multiplier = 2 if config.differential else 1
        self.head_dim = config.num_dims // self.num_heads
        self.betas = nn.Parameter(torch.ones(self.num_heads, 1, self.head_dim))
        self.init_states = nn.Parameter(
            torch.zeros(self.num_heads, self.head_dim * multiplier, self.head_dim)
        )
        self.init_z = nn.Parameter(
            torch.ones(self.num_heads, self.head_dim * multiplier) / self.head_dim
        )
        nn.init.kaiming_uniform_(self.init_states)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, output: Tensor):
        # Start with an initial state
        current_states, current_z = self.init_states, self.init_z
        # Blend with intermediate states
        memory_states, memory_z = self._compute_updates(
            key, value, current_states, current_z
        )
        # Retrieve using accumulated state
        memory_output = self._retrieve_memory(query, memory_states, memory_z)
        # Combine with attention
        return self._focus_attention(memory_output, output)

    def _retrieve_memory(self, query, memory_states, memory_z):
        # Retrieve using accumulated state
        sigma_q = F.elu(query) + 1.0
        retrieved_memories = torch.matmul(sigma_q, memory_states)
        norm_factor = torch.matmul(sigma_q, memory_z.unsqueeze(-1)) + self.epsilon
        return retrieved_memories / norm_factor

    def _compute_updates(self, key, value, current_states, current_z):
        # Compute memory updates
        sigma_k = F.elu(key) + 1.0
        if self.use_delta:
            retrieved_value = torch.matmul(sigma_k, current_states) / (
                torch.matmul(sigma_k, current_z.unsqueeze(-1)) + self.epsilon
            )
            delta_value = value - retrieved_value
            updates = current_states + torch.matmul(
                sigma_k.transpose(-2, -1), delta_value
            )
        else:
            updates = torch.matmul(sigma_k.transpose(-2, -1), value)
        z_updates = sigma_k.sum(dim=-2)
        # Accumulate states
        memory_states = current_states + updates
        memory_z = current_z + z_updates
        return memory_states, memory_z

    def _focus_attention(self, memory_output, attention_output):
        gate = torch.sigmoid(self.betas)
        return gate * memory_output + (1 - gate) * attention_output
