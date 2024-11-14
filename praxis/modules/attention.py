import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from praxis.modules.memory import PraxisMemory


class PraxisAttention(nn.Module):
    """
    This class is akin a wrapper, which implements a number of interesting attention
    mechanisms, and makes them optional with feature flags. By toggling features, one can
    essentially blend components from various kinds of attention.
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.causal = config.causal
        self.differential = config.differential
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

        # The core attention mechanism
        if self.differential:
            self.algorithm = Differential(config)
        else:
            self.algorithm = ScaledDotProduct(config)

        # For handling length extrapolation
        self.alibi = True
        if self.alibi:
            self.encoding = ALiBi(config)
        else:
            self.encoding = NoPE(config)

        # Add memory-related parameters
        self.use_memory = config.memory
        if self.use_memory:
            self.memory = PraxisMemory(config)

        # Standard output projection
        self.output = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self, inputs: Tensor, attention_mask: Tensor, token_indices: Optional[Tensor]
    ):
        batch_size, seq_len, _ = inputs.shape

        # Compute queries, keys, and values
        multiplier = (self.query.weight.size(0) // self.num_heads) // self.head_dim
        q = (
            self.query(inputs)
            .view(batch_size, -1, self.num_heads, self.head_dim * multiplier)
            .transpose(1, 2)
        )
        k = (
            self.key(inputs)
            .view(batch_size, -1, self.num_heads, self.head_dim * multiplier)
            .transpose(1, 2)
        )
        v = (
            self.value(inputs)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores
        scores = self.algorithm.compute_scores(q, k)

        # Compute positional encoding
        scores = self.encoding(scores, token_indices)

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
        weights = self.algorithm.compute_weights(scores)

        # Compute attention output
        outputs = self.algorithm.compute_outputs(weights, v)

        # Add memory-based attention
        if self.use_memory:
            outputs = self.memory(inputs, q, k, v, outputs)

        # Reshape for output projection
        outputs = outputs.transpose(1, 2).reshape(
            batch_size, seq_len, self.hidden_size
        )  # Shape: (batch_size, seq_len, num_heads * head_dim)

        # Output projection
        return self.output(outputs)


class ScaledDotProduct(nn.Module):
    """
    This class implements scaled dot-product attention:
    https://paperswithcode.com/method/scaled
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.hidden_size = config.num_dims
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        # Force exploration of attention subnetworks
        self.dropout = nn.Dropout(config.dropout)

    def compute_scores(self, q, k):
        reciprocal = 1.0 / math.sqrt(self.head_dim)
        scores = [torch.matmul(q, k.transpose(-2, -1)) * reciprocal]
        return scores

    def compute_weights(self, scores):
        weights = [self.dropout(F.softmax(score, dim=-1)) for score in scores]
        return weights[0]

    def compute_outputs(self, weights, v):
        # Compute attention output
        return weights @ v  # Shape: (batch_size, num_heads, seq_len, head_dim)


class Differential(ScaledDotProduct):
    """
    This class implements Differential Attention, to filter the noise from attention maps:
    https://arxiv.org/abs/2410.05258
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__(config)
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

    def compute_scores(self, q, k):
        # Split queries and keys
        Q1, Q2 = q[..., : self.head_dim], q[..., self.head_dim :]
        K1, K2 = k[..., : self.head_dim], k[..., self.head_dim :]
        # Compute differential attention scores
        reciprocal = 1.0 / math.sqrt(self.head_dim)
        scores = [
            torch.matmul(Q1, K1.transpose(-2, -1)) * reciprocal,
            torch.matmul(Q2, K2.transpose(-2, -1)) * reciprocal,
        ]
        return scores

    def compute_weights(self, scores):
        weights = [self.dropout(F.softmax(score, dim=-1)) for score in scores]
        # Compute scalar lambda
        lambda_scalar = (
            torch.exp(torch.dot(self.lambdas["q1"], self.lambdas["k1"]))
            - torch.exp(torch.dot(self.lambdas["q2"], self.lambdas["k2"]))
            + self.lambda_init
        )
        attention_weights = weights[0] - lambda_scalar * weights[1]
        return attention_weights

    def compute_outputs(self, weights, v):
        outputs = super().compute_outputs(weights, v)
        batch_size, num_heads, seq_len, head_dim = outputs.shape
        # Reshape for GroupNorm
        attention_output = (
            outputs.permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, num_heads * head_dim)
            .permute(0, 2, 1)
            .contiguous()
        )  # Shape: (batch_size, num_heads * head_dim, seq_len)
        # Apply GroupNorm
        attention_output = self.norm(attention_output)
        # Permute to original shape
        attention_output = (
            attention_output.permute(0, 2, 1)
            .view(batch_size, seq_len, num_heads, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # Shape: (batch_size, num_heads, seq_len, head_dim)
        # Apply scaling factor
        attention_output = attention_output * (1 - self.lambda_init)
        return attention_output


class NoPE(nn.Module):
    """
    This class is like nn.Identity(), because it implements no positional
    encoding at all.
    https://arxiv.org/abs/2404.12224
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()

    def forward(self, scores, token_indices):
        return scores


class ALiBi(NoPE):
    """
    This class implements Attention with Linear Biases (ALiBi), which is a form of
    length extrapolation that does not require trainable parameters.
    https://arxiv.org/abs/2108.12409
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        # Pre-compute the ALiBi slopes
        slopes = 2 ** (-8 * torch.arange(1, config.num_heads + 1) / config.num_heads)
        self.register_buffer("slopes", slopes)
        self.register_buffer(
            "positions", torch.arange(config.context_length, dtype=torch.float32)
        )

    def forward(self, scores, token_indices):
        batch_size, num_heads, seq_len, _ = scores[0].shape
        if torch.is_tensor(token_indices):
            # If token indices were provided (by a router, perhaps), use them
            positions = self.positions[token_indices]
        else:
            # Else, slice from the pre-computed ALiBi biases
            positions = (
                self.positions[:seq_len].unsqueeze(0).expand(batch_size, seq_len)
            )
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        biases = self.slopes.view(1, num_heads, 1, 1) * pos_diff.unsqueeze(1)
        scores = [score - biases for score in scores]
        return scores
