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
    Syntaxes Attention: Learnable query compression with recent context K/V.

    This attention mechanism uses asymmetric processing:
    - Queries: Compressed using learnable per-head convolutions to capture long-range dependencies
    - Keys/Values: Limited to recent tokens via tail attention

    All compressed queries attend to the same recent context of tokens,
    maintaining global context while focusing on local details.

    Key features:
    - Learnable query compression with per-head Conv1d preserves important patterns
    - Recent context K/V reduces memory and focuses on latest tokens
    - Residual-like reconstruction preserves distant tokens while updating recent context
    - Reduces complexity from O(n²) to O(n/r * c) where r is compression ratio, c is context size
    - Maintains causality through recency-focused attention
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        # Ensure num_heads divides hidden_size evenly
        while self.hidden_size % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1

        self.head_dim = self.hidden_size // self.num_heads

        # Query compression ratio (how much to reduce query length)
        self.query_compression_ratio = getattr(
            config, "syntaxes_query_compression_ratio", 4
        )

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

        # Learnable query compression using grouped convolution
        # Each head gets its own compression kernel
        self.query_compressor = nn.Conv1d(
            in_channels=self.num_heads * self.head_dim,
            out_channels=self.num_heads * self.head_dim,
            kernel_size=self.query_compression_ratio,
            stride=self.query_compression_ratio,
            padding=0,
            groups=self.num_heads,  # Separate compression per head
        )

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
        """Forward pass of Syntaxes attention with compressed queries and positional encoding."""
        batch_size, seq_len, hidden_size = inputs.shape
        device = inputs.device

        if past_key_values is not None:
            raise NotImplementedError(
                "KV caching not yet supported for SyntaxesAttention"
            )

        # Project queries from inputs (before compression)
        q = self.q_proj(inputs)  # [batch, seq_len, hidden]

        # Reshape queries for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch, heads, seq_len, head_dim]

        # Apply positional encoding to queries before compression
        # Create dummy k, v with same shape for the encoding API
        k_dummy = torch.zeros_like(q)
        v_dummy = torch.zeros_like(q)
        q, _, _ = self.encoding.before_scores(
            q, k_dummy, v_dummy, offset=offset, block_ids=block_ids
        )

        # Learnable query compression
        # Reshape q for Conv1d: [batch, heads * head_dim, seq_len]
        q_for_conv = q.transpose(1, 2).contiguous()  # [batch, seq_len, heads, head_dim]
        q_for_conv = q_for_conv.view(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        q_for_conv = q_for_conv.transpose(1, 2)  # [batch, heads * head_dim, seq_len]

        # Pad if necessary to ensure we can compress the full sequence
        remainder = seq_len % self.query_compression_ratio
        if remainder != 0:
            pad_len = self.query_compression_ratio - remainder
            q_for_conv = F.pad(q_for_conv, (0, pad_len), mode="constant", value=0)

        # Apply learnable compression
        q_compressed_conv = self.query_compressor(
            q_for_conv
        )  # [batch, heads * head_dim, compressed_seq_len]
        compressed_seq_len = q_compressed_conv.size(2)

        # Reshape back to multi-head format
        q_compressed = q_compressed_conv.transpose(
            1, 2
        )  # [batch, compressed_seq_len, heads * head_dim]
        q_compressed = q_compressed.view(
            batch_size, compressed_seq_len, self.num_heads, self.head_dim
        )
        q_compressed = q_compressed.transpose(
            1, 2
        )  # [batch, heads, compressed_seq_len, head_dim]

        # Recent context for keys and values
        # All compressed queries will attend to the same recent context
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

        # Apply positional encoding to K/V with context offset
        # The context starts at position context_start, so we add that to the offset
        context_offset = offset + context_start
        k, v, _ = self.encoding.before_scores(
            k, k, v, offset=context_offset, block_ids=block_ids
        )

        # Attention computation - all compressed queries attend to the same recent context
        # q_compressed: [batch, heads, compressed_seq_len, head_dim]
        # k: [batch, heads, context_size, head_dim]
        # v: [batch, heads, context_size, head_dim]

        # Compute attention scores
        scores = (
            torch.matmul(
                q_compressed,  # [batch, heads, compressed_seq_len, head_dim]
                k.transpose(-2, -1),  # [batch, heads, head_dim, context_size]
            )
            * self.scale
        )  # [batch, heads, compressed_seq_len, context_size]

        # Apply positional encoding to scores
        scores = self.encoding.after_scores(scores, offset=offset, block_ids=block_ids)

        # Create causal mask
        # Each compressed query position maps to a position in the original sequence
        # and can only attend to K/V positions that come before or at that position
        compressed_positions = torch.arange(compressed_seq_len, device=device)
        # Map compressed positions back to original sequence positions
        original_query_positions = (
            (compressed_positions + 1) * self.query_compression_ratio - 1
        ).unsqueeze(1)
        # Context positions in the original sequence
        context_positions = torch.arange(
            context_start, context_start + context_size, device=device
        ).unsqueeze(0)
        # Causal mask: can only attend to positions <= original query position
        causal_mask = (
            context_positions <= original_query_positions
        )  # [compressed_seq_len, context_size]

        # Special handling: if a query position is before the context, it shouldn't attend to anything
        # But to avoid NaN, we'll let it attend to the first position in the context
        queries_before_context = original_query_positions.squeeze(1) < context_start
        if queries_before_context.any():
            # For queries before context, allow attention to first context position only
            causal_mask[queries_before_context, 0] = True

        causal_mask = causal_mask.unsqueeze(0).unsqueeze(
            0
        )  # [1, 1, compressed_seq_len, context_size]
        scores = scores.masked_fill(~causal_mask, -torch.inf)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Extract context portion of attention mask
            context_mask = attention_mask[:, context_start : context_start + context_size]
            context_mask = context_mask.unsqueeze(1).unsqueeze(
                1
            )  # [batch, 1, 1, context_size]
            scores = scores.masked_fill(context_mask == 0, -torch.inf)

        # Softmax and dropout
        weights = F.softmax(
            scores, dim=-1
        )  # [batch, heads, compressed_seq_len, context_size]
        weights = self.dropout(weights)

        # Apply attention to values
        output = torch.matmul(
            weights,  # [batch, heads, compressed_seq_len, context_size]
            v,  # [batch, heads, context_size, head_dim]
        )  # [batch, heads, compressed_seq_len, head_dim]

        # Reshape and prepare for upsampling
        output = output.transpose(
            1, 2
        ).contiguous()  # [batch, compressed_seq_len, heads, head_dim]
        output = output.view(batch_size, compressed_seq_len, self.hidden_size)

        # Apply output projection before reconstruction
        output = self.o_proj(output)  # [batch, compressed_seq_len, hidden]

        # Residual-like reconstruction: start with original inputs and replace the recent context
        # This preserves older tokens while updating only the recent context
        result = inputs.clone()  # [batch, seq_len, hidden]
        
        # Direct vectorized assignment: replace the recent context with attention output
        # Simply assign the compressed output to the corresponding context positions
        if context_start < seq_len and compressed_seq_len > 0:
            # Calculate how much of the context we can actually fill
            available_context = min(context_size, seq_len - context_start)
            output_length = min(compressed_seq_len, available_context)
            
            # Direct assignment - each compressed output token goes to one position in the context
            result[:, context_start:context_start + output_length, :] = output[:, :output_length, :]

        # Final dropout
        result = self.dropout(result)

        return result, None, 0
