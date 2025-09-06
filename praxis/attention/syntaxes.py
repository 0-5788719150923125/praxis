import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import DynamicCache

from praxis.encoding import ENCODING_REGISTRY


class SyntaxesAttention(nn.Module):
    """
    Syntaxes Attention: All queries attend to reduced K/V context.

    This attention mechanism uses asymmetric processing:
    - Queries: All positions in the sequence generate queries
    - Keys/Values: Limited to recent tokens for memory efficiency

    Every query position attends to the same reduced context,
    focusing computation on the most relevant recent information.

    Key features:
    - Full query coverage preserves all positional information
    - Reduced K/V context focuses on recent tokens
    - Simple and efficient - no compression/reconstruction needed
    - Reduces complexity from O(n²) to O(n * c) where c is context size
    - Maintains causality through recent context attention
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        # Ensure num_heads divides hidden_size evenly
        while self.hidden_size % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1

        self.head_dim = self.hidden_size // self.num_heads

        # Recent context size for keys/values
        self.context_size = getattr(config, "syntaxes_context_size", 128)

        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Dropout for regularization
        self.dropout = nn.Dropout(getattr(config, "dropout", 0.1))

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Create a modified config for the encoding that reflects the actual number of heads
        encoding_config = type(config)(**vars(config))
        encoding_config.num_heads = self.num_heads
        # Keep num_queries as 1 to maintain the same total query heads
        encoding_config.num_queries = 1

        # Positional encoding
        self.encoding = ENCODING_REGISTRY[config.encoding](encoding_config)

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Union[Tensor, DynamicCache]] = None,
        block_ids: Optional[Tensor] = None,
        current_depth: int = 0,
        offset: int = 0,
    ) -> Tuple[Tensor, Optional[Union[Tensor, DynamicCache]], Union[int, float]]:
        """Forward pass of Syntaxes attention - all queries attend to recent context K/V."""
        batch_size, seq_len, hidden_size = inputs.shape
        device = inputs.device

        if past_key_values is not None:
            raise NotImplementedError(
                "KV caching not yet supported for SyntaxesAttention"
            )

        # Project ALL queries from ALL inputs
        q = self.q_proj(inputs)  # [batch, seq_len, hidden]

        # Reshape queries for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch, heads, seq_len, head_dim]

        # Recent context for keys and values
        context_size = min(self.context_size, seq_len)
        context_start = max(0, seq_len - context_size)

        # Extract the recent context tokens for K/V
        context_inputs = inputs[:, context_start:, :]  # [batch, context_size, hidden]

        # Project K/V from the recent context
        k = self.k_proj(context_inputs)  # [batch, context_size, hidden]
        v = self.v_proj(context_inputs)  # [batch, context_size, hidden]

        # Reshape K/V for multi-head attention
        k = k.view(batch_size, context_size, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch, heads, context_size, head_dim]
        v = v.view(batch_size, context_size, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch, heads, context_size, head_dim]

        # Apply positional encoding
        context_offset = offset + context_start
        q, _, _ = self.encoding.before_scores(
            q, q, q, offset=offset, block_ids=block_ids
        )
        k, v, _ = self.encoding.before_scores(
            k, k, v, offset=context_offset, block_ids=block_ids
        )

        # Attention computation - ALL queries attend to the same recent context
        # q: [batch, heads, seq_len, head_dim]
        # k: [batch, heads, context_size, head_dim]
        # v: [batch, heads, context_size, head_dim]

        # Compute attention scores
        scores = (
            torch.matmul(
                q,  # [batch, heads, seq_len, head_dim]
                k.transpose(-2, -1),  # [batch, heads, head_dim, context_size]
            )
            * self.scale
        )  # [batch, heads, seq_len, context_size]

        # Apply positional encoding to scores
        scores = self.encoding.after_scores(scores, offset=offset, block_ids=block_ids)

        # Create causal mask - each query can only attend to context positions <= its own position
        query_positions = torch.arange(seq_len, device=device).unsqueeze(
            1
        )  # [seq_len, 1]
        context_positions = torch.arange(
            context_start, context_start + context_size, device=device
        ).unsqueeze(
            0
        )  # [1, context_size]

        # Causal mask: can only attend to positions <= query position
        causal_mask = context_positions <= query_positions  # [seq_len, context_size]

        # Special handling: queries before the context window need at least one valid attention target
        # to avoid NaN. We'll let them attend to the first position in the context.
        queries_before_context = query_positions.squeeze(1) < context_start
        if queries_before_context.any():
            # Allow these queries to attend to the first context position
            causal_mask[queries_before_context, 0] = True

        causal_mask = causal_mask.unsqueeze(0).unsqueeze(
            0
        )  # [1, 1, seq_len, context_size]
        scores = scores.masked_fill(~causal_mask, -torch.inf)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Extract context portion of attention mask
            context_mask = attention_mask[
                :, context_start : context_start + context_size
            ]
            context_mask = context_mask.unsqueeze(1).unsqueeze(
                1
            )  # [batch, 1, 1, context_size]
            scores = scores.masked_fill(context_mask == 0, -torch.inf)

        # Softmax and dropout
        weights = F.softmax(scores, dim=-1)  # [batch, heads, seq_len, context_size]
        weights = self.dropout(weights)

        # Apply attention to values
        output = torch.matmul(
            weights,  # [batch, heads, seq_len, context_size]
            v,  # [batch, heads, context_size, head_dim]
        )  # [batch, heads, seq_len, head_dim]

        # Reshape output
        output = output.transpose(
            1, 2
        ).contiguous()  # [batch, seq_len, heads, head_dim]
        output = output.view(batch_size, seq_len, self.hidden_size)

        # Apply output projection
        output = self.o_proj(output)  # [batch, seq_len, hidden]

        # Final dropout
        output = self.dropout(output)

        return output, None, 0
