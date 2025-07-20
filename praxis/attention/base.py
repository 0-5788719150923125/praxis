import math
from copy import copy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import DynamicCache

from praxis.attention.components import (
    GatedEMA,
    MultiTokenAttention,
    UniversalAttentionGate,
)
from praxis.attention.core import Differential, Linear, ScaledDotProduct, Stickbreaking
from praxis.attention.memory import CompressiveMemory
from praxis.attention.mla import MLAKeyValue, MLAQuery
from praxis.attention.pk_attention import ProductKeyAttention
from praxis.attention.projections import LinearKeyValue, LinearQuery, LowRankKeyValue
from praxis.attention.sparse_query import SparseQuery
from praxis.attention.thc import TemporalHealthComplex
from praxis.dense import DENSE_REGISTRY
from praxis.encoding import ENCODING_REGISTRY

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class ModularAttention(nn.Module):
    """
    This class is akin to a wrapper, which implements a number of interesting attention
    mechanisms, and makes them optional with feature flags. By toggling features, one can
    essentially blend components from various kinds of attention.
    """

    __version__ = "0.1.0"

    def __init__(self, config: ConfigType) -> None:
        """
        Initialize ModularAttention module with configuration.

        Args:
            config: Configuration object containing attention parameters
        """
        super().__init__()
        self.causal: bool = config.causal
        # Set the core attention mechanism
        self.linear: bool = config.linear
        self.differential: bool = config.differential
        self.stickbreaking: bool = config.stickbreaking
        self.mla: bool = config.mla
        assert (
            sum([self.differential, self.stickbreaking, self.linear]) <= 1
        ), "Only one of differential, stickbreaking, or linear attention can be used at a time."

        hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_heads
        self.num_queries: int = config.num_queries
        self.num_query_heads: int = self.num_heads * self.num_queries
        self.kv_rank: Optional[int] = config.kv_rank

        self.factor: int = 2 if self.differential else 1
        self.head_dim: int = (
            getattr(config, "head_size") or hidden_size // self.num_heads // self.factor
        )
        setattr(config, "head_size", self.head_dim)

        # MLA-specific dimensions
        if self.mla:
            # Calculate compression dimensions automatically
            self.kv_compression_dim: int = hidden_size // 8
            self.q_compression_dim: int = hidden_size // 8
            # Decoupled RoPE dimensions
            self.rope_head_dim: int = self.head_dim // 4
            self.use_mla_in_inference: bool = True

            # Set these on the config for MLA modules to access them
            setattr(config, "kv_compression_dim", self.kv_compression_dim)
            setattr(config, "q_compression_dim", self.q_compression_dim)
            setattr(config, "rope_head_dim", self.rope_head_dim)

        assert (
            sum([config.mega, config.gated]) <= 1
        ), "Only one of 'mega' or 'gated' can be used at a time."

        # For query gating
        self.ema: Union[GatedEMA, bool] = GatedEMA(config) if config.mega else False

        # Query and key projections for differential heads
        if self.mla:
            # MLA uses compressed projections followed by up-projections
            self.query = MLAQuery(config)
        elif config.k_heads is not None:
            self.query = SparseQuery(
                hidden_size,
                self.num_query_heads,
                self.head_dim * self.factor,
                top_k=config.k_heads,
                dropout=config.dropout,
                bias=False,
                debug=config.debug,
            )
        else:
            self.query = LinearQuery(
                hidden_size,
                self.num_query_heads * self.head_dim * self.factor,
                bias=False,
            )

        if self.mla:
            self.key_value = MLAKeyValue(config)
        elif self.kv_rank is not None:
            self.key_value = LowRankKeyValue(
                hidden_size=hidden_size,
                num_heads=self.num_heads,
                key_head_dim=self.head_dim * self.factor,
                value_head_dim=self.head_dim,
                rank=self.kv_rank,
            )
        else:
            self.key_value = LinearKeyValue(
                hidden_size=hidden_size,
                num_heads=self.num_heads,
                key_head_dim=self.head_dim * self.factor,
                value_head_dim=self.head_dim,
            )

        self.memory: Union[CompressiveMemory, bool] = config.memory
        self.chunk_size: int = 0
        if self.memory:
            self.chunk_size = 256
            self.memory = CompressiveMemory(config)

        # The core attention mechanism
        if self.stickbreaking:
            self.core = Stickbreaking(config)
        elif self.differential:
            self.core = Differential(config)
        elif self.linear:
            self.core = Linear(config)
        else:
            self.core = ScaledDotProduct(config)

        # For handling length extrapolation
        self.encoding = ENCODING_REGISTRY[config.encoding](config)

        # For Multi-Token Attention
        self.mta: Union[MultiTokenAttention, bool] = (
            MultiTokenAttention(config) if config.mta else False
        )

        # For attention gating
        self.gates: Union[UniversalAttentionGate, bool] = (
            UniversalAttentionGate(config) if config.gated else False
        )

        # Standard output projection
        self.output = nn.Linear(
            self.num_query_heads * self.head_dim, hidden_size, bias=False
        )

        # Temporal Health Complex module for K/V enhancement
        self.thc_k = (
            TemporalHealthComplex(
                d_model=hidden_size,
                reduction_factor=8,
                kernel_size=3,
                dropout=config.dropout,
                gate_init="zeros",
            )
            if "use_thc" in config.meta
            else nn.Identity()
        )
        self.thc_v = (
            TemporalHealthComplex(
                d_model=hidden_size,
                reduction_factor=8,
                kernel_size=3,
                dropout=config.dropout,
                gate_init="zeros",
            )
            if "use_thc" in config.meta
            else nn.Identity()
        )

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Union[Tensor, DynamicCache]] = None,
        block_ids: Optional[Tensor] = None,
        current_depth: int = 0,
    ) -> Tuple[Tensor, Optional[Union[Tensor, DynamicCache]], Union[int, float]]:
        """
        Forward pass of the attention module.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask tensor for padding tokens
            past_key_values: Optional cache for key/value pairs from previous steps
            block_ids: Optional tensor indicating block structure for blocked attention
            current_depth: Current depth in the network (for caching)

        Returns:
            Tuple containing:
            - Output tensor after attention and projection
            - Updated cache (if using caching)
            - Auxiliary loss value
        """
        batch_size, seq_len, _ = inputs.shape
        aux_loss: Union[int, float] = 0

        if self.ema:
            # Compute an exponential moving average-based gating mechanism
            inputs = self.ema(inputs)

        # Initialize QKV projections
        q, aux_loss = self.query(inputs)

        # Apply THC to K/V projections
        k, v = self.key_value((self.thc_k(inputs), self.thc_v(inputs)))

        # Define the views
        q_view = (batch_size, seq_len, self.num_query_heads * self.factor, -1)
        k_view = (batch_size, seq_len, self.num_heads * self.factor, -1)
        v_view = (batch_size, seq_len, self.num_heads, -1)

        # Create the view and transpose
        q = q.view(q_view).transpose(1, 2)  # [b, h, s, d]
        k = k.view(k_view).transpose(1, 2)  # [b, h, s, d]
        v = v.view(v_view).transpose(1, 2)  # [b, h, s, d]

        # Handle KV caching
        if isinstance(past_key_values, DynamicCache):
            k, v = past_key_values.update(k, v, current_depth)
            full_seq_len = k.size(2)  # Get actual sequence length after cache
        else:
            full_seq_len = seq_len

        # Handle GQA (Grouped Query Attention)
        if self.num_queries > 1:
            k = k.repeat_interleave(self.num_queries, dim=1)
            v = v.repeat_interleave(self.num_queries, dim=1)

        # Determine chunk sizes based on whether we're using cache
        chunk_size = self.chunk_size if self.chunk_size > 0 else full_seq_len
        num_chunks = (full_seq_len + chunk_size - 1) // chunk_size

        outputs: List[Tensor] = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, full_seq_len)

            # During inference with cache:
            if isinstance(past_key_values, DynamicCache):
                chunk_q = q  # Take all of q (length 1)
            else:
                # Training/no-cache behavior remains the same
                chunk_q = q[:, :, start_idx:end_idx]

            chunk_k = k[:, :, start_idx:end_idx]
            chunk_v = v[:, :, start_idx:end_idx]

            chunk_mask: Optional[Tensor] = (
                None if attention_mask is None else attention_mask[:, start_idx:end_idx]
            )
            chunk_block_ids: Optional[Tensor] = None
            if block_ids is not None:
                if isinstance(past_key_values, DynamicCache):
                    chunk_block_ids = block_ids  # Keep full block_ids
                else:
                    chunk_block_ids = block_ids[:, start_idx:end_idx]
                if chunk_block_ids.dim() == 3:
                    chunk_block_ids = chunk_block_ids.squeeze(-1)

            current_chunk_size = (
                end_idx - start_idx
                if not isinstance(past_key_values, DynamicCache)
                else 1
            )

            # Process chunk with position offset
            chunk_output = self._process_chunk(
                chunk_q,
                chunk_k,
                chunk_v,
                chunk_mask,
                current_chunk_size,
                start_idx,
                chunk_block_ids,
            )

            outputs.append(chunk_output)

        # Concatenate all chunks
        output = torch.cat(outputs, dim=1)

        if self.memory:
            self.memory.reset_states()

        if self.gates:
            output = self.gates(inputs, output)

        # Final output projection
        return self.output(output), past_key_values, aux_loss

    def _process_chunk(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor],
        chunk_size: int,
        offset: int = 0,
        block_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Process a chunk of the attention computation.

        Args:
            q: Query tensor of shape [batch_size, num_heads, chunk_size, head_dim]
            k: Key tensor of shape [batch_size, num_heads, chunk_size, head_dim]
            v: Value tensor of shape [batch_size, num_heads, chunk_size, head_dim]
            attention_mask: Optional mask tensor for padding tokens
            chunk_size: Size of the current chunk
            offset: Position offset for positional encoding
            block_ids: Optional tensor indicating block structure

        Returns:
            Processed chunk output tensor
        """
        batch_size = q.size(0)

        if self.mla:
            # MLA uses decoupled RoPE - split the queries and keys into compressed and RoPE parts
            q_c = q[..., : self.head_dim]  # Compressed queries
            q_r = q[..., self.head_dim :]  # RoPE queries
            k_c = k[..., : self.head_dim]  # Compressed keys
            k_r = k[..., self.head_dim :]  # RoPE keys

            # Apply RoPE only to the RoPE components
            q_r, k_r, _ = self.encoding.before_scores(
                q_r, k_r, None, offset=offset, block_ids=block_ids
            )

            # Concatenate back
            q = torch.cat([q_c, q_r], dim=-1)
            k = torch.cat([k_c, k_r], dim=-1)
        else:
            # Apply positional encoding with offset
            q, k, v = self.encoding.before_scores(
                q, k, v, offset=offset, block_ids=block_ids
            )

        # Compute attention scores
        q, k, v, scores = self.core.compute_scores(q, k, v)
        hist_len = k.size(2)

        if self.mta:
            scores = self.mta.key_query_convolution(scores)

        # Apply positional encoding to scores
        scores = self.encoding.after_scores(scores, offset=offset, block_ids=block_ids)

        # Apply masking
        scores, causal_mask, chunk_attention_mask = self.core.apply_masking(
            scores, attention_mask, block_ids, chunk_size, hist_len, self.causal
        )

        # Compute the attention weights
        weights, v = self.core.compute_weights(
            q, k, v, scores, causal_mask, chunk_attention_mask
        )

        # Apply head mixing to the attention weights (post-softmax)
        if self.mta:
            weights = self.mta.head_mixing_convolution(weights)

        # Get attention output
        attention_output = self.core.compute_outputs(weights, v)

        if self.mta:
            attention_output = self.mta.group_norm(attention_output)

        if self.memory:
            # Blend with memories
            attention_output = self.memory(q, k, v, attention_output)

        # Reshape for output projection
        chunk_output = attention_output.transpose(1, 2).reshape(
            batch_size, chunk_size, -1
        )

        return chunk_output
