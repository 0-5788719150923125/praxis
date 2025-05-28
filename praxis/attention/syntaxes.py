import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import DynamicCache


class SyntaxesAttention(nn.Module):
    """
    Efficient Causal Syntaxes Attention with Sliding Window.
    
    Each position attends to the most recent k tokens using a causal sliding window.
    This reduces complexity from O(n²) to O(n*k) where k << n while preserving causality.
    
    Key features:
    - Fully vectorized implementation (no loops over sequence length)
    - Causal sliding window pattern for autoregressive models
    - No importance scoring overhead - uses simple recency-based selection
    - Maintains exact same interface as standard attention
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        # Ensure num_heads divides hidden_size evenly
        while self.hidden_size % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1

        self.head_dim = self.hidden_size // self.num_heads

        # Number of tokens to select for sliding window
        self.num_selected = getattr(config, "syntaxes_num_selected", 128)

        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Dropout for regularization
        self.dropout = nn.Dropout(getattr(config, "dropout", 0.1))

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Union[Tensor, DynamicCache]] = None,
        block_ids: Optional[Tensor] = None,
        current_depth: int = 0,
    ) -> Tuple[Tensor, Optional[Union[Tensor, DynamicCache]], Union[int, float]]:
        """Forward pass of causal Syntaxes attention with sliding window."""
        batch_size, seq_len, hidden_size = inputs.shape
        device = inputs.device

        if past_key_values is not None:
            raise NotImplementedError(
                "KV caching not yet supported for SyntaxesAttention"
            )

        # Causal sliding window approach
        # Create sliding window indices vectorized for all positions at once
        # Each position i attends to positions [max(0, i+1-num_selected), i]
        position_indices = torch.arange(seq_len, device=device)  # [seq_len]

        # For each position, create its sliding window
        # Using broadcasting to create all windows at once
        window_starts = torch.clamp(
            position_indices + 1 - self.num_selected, min=0
        )  # [seq_len]
        window_offsets = torch.arange(
            self.num_selected, device=device
        )  # [num_selected]

        # Create indices matrix: [seq_len, num_selected]
        selected_indices = window_starts.unsqueeze(1) + window_offsets.unsqueeze(
            0
        )  # [seq_len, num_selected]

        # Clamp to valid range and create causal mask
        selected_indices = torch.clamp(selected_indices, 0, seq_len - 1)

        # Create causal mask: position i can only attend to positions <= i
        query_positions = position_indices.unsqueeze(1)  # [seq_len, 1]
        causal_mask = selected_indices <= query_positions  # [seq_len, num_selected]

        # Expand for batch dimension
        selected_indices = selected_indices.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, seq_len, num_selected]
        causal_mask = causal_mask.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, seq_len, num_selected]

        # Gather tokens using advanced indexing
        batch_indices = torch.arange(batch_size, device=device).view(
            batch_size, 1, 1
        )
        gathered_tokens = inputs[
            batch_indices, selected_indices
        ]  # [batch, seq_len, num_selected, hidden]

        # Project all at once
        q = self.q_proj(inputs)  # [batch, seq_len, hidden]
        k = self.k_proj(gathered_tokens)  # [batch, seq_len, num_selected, hidden]
        v = self.v_proj(gathered_tokens)  # [batch, seq_len, num_selected, hidden]

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch, heads, seq_len, head_dim]
        k = k.view(
            batch_size, seq_len, self.num_selected, self.num_heads, self.head_dim
        ).permute(
            0, 3, 1, 2, 4
        )  # [batch, heads, seq_len, num_selected, head_dim]
        v = v.view(
            batch_size, seq_len, self.num_selected, self.num_heads, self.head_dim
        ).permute(
            0, 3, 1, 2, 4
        )  # [batch, heads, seq_len, num_selected, head_dim]

        # Vectorized attention computation
        # Compute all attention scores at once
        scores = (
            torch.matmul(
                q.unsqueeze(-2),  # [batch, heads, seq_len, 1, head_dim]
                k.transpose(
                    -2, -1
                ),  # [batch, heads, seq_len, head_dim, num_selected]
            ).squeeze(-2)
            * self.scale
        )  # [batch, heads, seq_len, num_selected]

        # Apply causal mask
        causal_mask_expanded = causal_mask.unsqueeze(1).expand(
            -1, self.num_heads, -1, -1
        )
        scores = scores.masked_fill(~causal_mask_expanded, -torch.inf)

        # Apply attention mask if provided
        if attention_mask is not None:
            query_mask = attention_mask.unsqueeze(1).unsqueeze(
                -1
            )  # [batch, 1, seq_len, 1]
            scores = scores.masked_fill(query_mask == 0, -torch.inf)

        # Softmax and dropout
        weights = F.softmax(scores, dim=-1)  # [batch, heads, seq_len, num_selected]
        weights = self.dropout(weights)

        # Apply attention to values - vectorized
        output = torch.matmul(
            weights.unsqueeze(-2),  # [batch, heads, seq_len, 1, num_selected]
            v,  # [batch, heads, seq_len, num_selected, head_dim]
        ).squeeze(
            -2
        )  # [batch, heads, seq_len, head_dim]

        # Reshape back
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        )

        # Final projection
        output = self.o_proj(output)
        output = self.dropout(output)

        return output, None, 0
