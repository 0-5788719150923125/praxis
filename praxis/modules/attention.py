import math
from copy import copy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import DynamicCache

from praxis.dense import DENSE_REGISTRY
from praxis.functional import alpha_entmax, alpha_relu, ghostmax
from praxis.modules.encoding import ENCODING_REGISTRY
from praxis.modules.experimental.pk_attention import ProductKeyAttention
from praxis.modules.experimental.sparse_query import SparseQuery
from praxis.modules.memory import PraxisCompressiveMemory


class PraxisAttention(nn.Module):
    """
    This class is akin to a wrapper, which implements a number of interesting attention
    mechanisms, and makes them optional with feature flags. By toggling features, one can
    essentially blend components from various kinds of attention.
    """

    __version__ = "0.1.0"

    def __init__(self, config: Any) -> None:
        """
        Initialize PraxisAttention module with configuration.

        Args:
            config: Configuration object containing attention parameters
        """
        super().__init__()
        self.causal: bool = config.causal
        # Set the core attention mechanism
        self.linear: bool = config.linear
        self.differential: bool = config.differential
        self.stickbreaking: bool = config.stickbreaking
        assert (
            sum([self.differential, self.stickbreaking, self.linear]) <= 1
        ), "Only one of differential, stickbreaking, or linear attention can be used at a time."

        hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_heads
        self.num_queries: int = config.num_queries
        self.num_query_heads: int = self.num_heads * self.num_queries
        self.kv_rank: Optional[int] = config.kv_rank

        self.factor: int = 2 if self.differential else 1
        self.head_dim: int = hidden_size // self.num_heads // self.factor
        setattr(config, "head_size", self.head_dim)

        assert (
            sum([config.mega, config.gated]) <= 1
        ), "Only one of 'mega' or 'gated' can be used at a time."

        # For query gating
        self.ema: Union[PraxisGatedEMA, bool] = (
            PraxisGatedEMA(config) if config.mega else False
        )

        # Query and key projections for differential heads
        if config.k_heads is not None:
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

        if self.kv_rank is not None:
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

        self.memory: Union[PraxisCompressiveMemory, bool] = config.memory
        self.chunk_size: int = 0
        if self.memory:
            self.chunk_size = 256
            self.memory = PraxisCompressiveMemory(config)

        # The core attention mechanism
        if self.stickbreaking:
            self.algorithm = Stickbreaking(config)
        elif self.differential:
            self.algorithm = Differential(config)
        elif self.linear:
            self.algorithm = Linear(config)
        else:
            self.algorithm = ScaledDotProduct(config)

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
        k, v = self.key_value(inputs)

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

        # Apply positional encoding with offset
        q, k, v = self.encoding.before_scores(
            q, k, v, offset=offset, block_ids=block_ids
        )

        # Compute attention scores
        q, k, v, scores = self.algorithm.compute_scores(q, k, v)
        hist_len = k.size(2)

        if self.mta:
            scores = self.mta.key_query_convolution(scores)

        # Apply positional encoding to scores
        scores = self.encoding.after_scores(scores, offset=offset, block_ids=block_ids)

        # Apply masking
        scores, causal_mask, chunk_attention_mask = self.algorithm.apply_masking(
            scores, attention_mask, block_ids, chunk_size, hist_len, self.causal
        )

        # Compute the attention weights
        weights, v = self.algorithm.compute_weights(
            q, k, v, scores, causal_mask, chunk_attention_mask
        )

        # Apply head mixing to the attention weights (post-softmax)
        if self.mta:
            weights = self.mta.head_mixing_convolution(weights)

        # Get attention output
        attention_output = self.algorithm.compute_outputs(weights, v)

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


class ScaledDotProduct(nn.Module):
    """
    This class implements scaled dot-product attention:
    https://paperswithcode.com/method/scaled
    """

    __version__ = "0.1.0"

    def __init__(self, config: Any) -> None:
        """
        Initialize scaled dot-product attention module.

        Args:
            config: Configuration object containing attention parameters
        """
        super().__init__()
        self.causal: bool = config.causal
        self.meta: str = config.meta
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_heads
        self.num_query_heads: int = self.num_heads * config.num_queries
        self.head_dim: int = config.head_size
        # Force exploration of attention subnetworks
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def compute_scores(
        self, q: Tensor, k: Tensor, v: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute attention scores between queries and keys.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor

        Returns:
            Tuple of (query, key, value, attention scores)
        """
        scaling = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
        return q, k, v, scores

    def apply_masking(
        self,
        scores: Tensor,
        attention_mask: Optional[Tensor],
        block_ids: Optional[Tensor],
        seq_len: int,
        hist_len: int,
        causal: bool,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Apply masking to attention scores.

        Args:
            scores: Attention scores
            attention_mask: Optional padding mask
            block_ids: Optional block structure tensor
            seq_len: Current sequence length
            hist_len: History length
            causal: Whether to apply causal masking

        Returns:
            Tuple of (masked scores, causal mask, attention mask)
        """
        causal_mask: Optional[Tensor] = None
        # When using caching
        if scores.size(2) == 1:
            return scores, causal_mask, attention_mask
        if causal:
            if block_ids is None:
                # Regular causal mask when no sequence blocking needed
                causal_mask = (
                    torch.triu(
                        torch.full((seq_len, hist_len), -1e9, device=scores.device),
                        diagonal=1,
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
            else:
                # Create block diagonal causal mask
                same_block = block_ids.unsqueeze(-1) == block_ids[
                    ..., :hist_len
                ].unsqueeze(-2)
                pos = torch.arange(seq_len, device=scores.device)
                causal = pos.unsqueeze(-1) >= torch.arange(
                    hist_len, device=scores.device
                )

                mask = (same_block & causal).unsqueeze(1).float()
                causal_mask = (1.0 - mask) * -1e9

            scores = scores + causal_mask
        elif attention_mask is not None:
            # Handle padding mask
            attention_mask = F.pad(
                attention_mask, (hist_len - attention_mask.size(-1), 0), value=1
            )
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9

            scores = scores + attention_mask

        return scores, causal_mask, attention_mask

    def compute_weights(
        self, q: Tensor, k: Tensor, v: Tensor, scores: Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute attention weights from scores.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            scores: Attention scores

        Returns:
            Tuple of (attention weights, value tensor)
        """
        if "entmax" in self.meta:
            weights = alpha_entmax(scores, dim=-1)
        elif "relu" in self.meta:
            weights = alpha_relu(scores, dim=-1, alpha=1.5, tau=None)
        elif "softmax" in self.meta:
            weights = F.softmax(scores, dim=-1)
        else:
            weights = ghostmax(scores, dim=-1)
        return weights, v

    def compute_outputs(self, weights: Tensor, v: Tensor) -> Tensor:
        """
        Compute final attention outputs.

        Args:
            weights: Attention weights
            v: Value tensor

        Returns:
            Output tensor after attention
        """
        return self.dropout(weights) @ v


class Differential(ScaledDotProduct):
    """
    This class implements Differential Attention, to filter the noise from attention maps:
    https://arxiv.org/abs/2410.05258
    """

    __version__ = "0.1.0"

    def __init__(self, config: Any) -> None:
        """
        Initialize differential attention module.

        Args:
            config: Configuration object containing attention parameters
        """
        super().__init__(config)
        self.lambda_init: float = 0.8
        head_dim: int = config.head_size

        # Parameters for differential attention
        self.lambda_q1: nn.Parameter = nn.Parameter(
            torch.zeros(head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k1: nn.Parameter = nn.Parameter(
            torch.zeros(head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_q2: nn.Parameter = nn.Parameter(
            torch.zeros(head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k2: nn.Parameter = nn.Parameter(
            torch.zeros(head_dim).normal_(mean=0, std=0.1)
        )

        # GroupNorm should match the total channels
        self.norm: nn.GroupNorm = nn.GroupNorm(
            num_groups=self.num_query_heads,
            num_channels=self.num_query_heads * head_dim,
            eps=config.epsilon,
        )

    def compute_weights(
        self, q: Tensor, k: Tensor, v: Tensor, scores: Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute differential attention weights from scores.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            scores: Attention scores

        Returns:
            Tuple of (attention weights, value tensor)
        """
        batch_size, num_heads, seq_len, _ = scores.shape
        head_dim = self.head_dim

        # Keep original attention shape, just split the heads
        attn_weights = ghostmax(scores, dim=-1)
        # Shape: [batch_size, num_heads, seq_len, seq_len]

        # Reshape to separate differential components
        attn_weights = attn_weights.view(batch_size, -1, 2, seq_len, seq_len)
        # Shape: [batch_size, num_heads, 2, seq_len, seq_len]

        # Calculate lambda
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Apply differential attention
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        # Shape: [batch_size, num_heads, seq_len, seq_len]

        return attn_weights, v

    def compute_outputs(self, weights: Tensor, v: Tensor) -> Tensor:
        """
        Compute final differential attention outputs with normalization.

        Args:
            weights: Attention weights
            v: Value tensor

        Returns:
            Output tensor after differential attention
        """
        batch_size, num_heads, seq_len, _ = weights.shape

        # Apply attention to values
        outputs = torch.matmul(self.dropout(weights), v)
        # Shape: [batch_size, num_heads, seq_len, head_dim]

        # Prepare for GroupNorm
        outputs = outputs.reshape(batch_size, -1, seq_len).contiguous()
        # Shape: [batch_size, seq_len, num_heads * head_dim]

        # Apply GroupNorm
        outputs = self.norm(outputs)
        # Shape: [batch_size, seq_len, num_heads * head_dim]

        return outputs * (1 - self.lambda_init)


class Linear(ScaledDotProduct):
    """
    Implements Linear Attention with efficient operations.
    Based on kernelized attention patterns which have O(L) instead of O(L²) complexity.
    """

    __version__ = "0.1.0"

    def __init__(self, config: Any) -> None:
        """
        Initialize linear attention module.

        Args:
            config: Configuration object containing attention parameters
        """
        super().__init__(config)
        self.epsilon = 1e-6
        # Optional normalization layer to stabilize computations
        self.norm = nn.LayerNorm(config.head_size, eps=config.epsilon)

    def compute_scores(
        self, q: Tensor, k: Tensor, v: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Apply feature map to queries and keys.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor

        Returns:
            Tuple of (mapped query, mapped key, value, placeholder)
        """
        # Use elu(x) + 1 as feature map to ensure positivity
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # We don't actually use the scores in linear attention
        placeholder = torch.tensor(0.0)

        return q, k, v, placeholder

    def apply_masking(
        self,
        scores: Tensor,
        attention_mask: Optional[Tensor],
        block_ids: Optional[Tensor],
        seq_len: int,
        hist_len: int,
        causal: bool,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Pass-through for masking (handled in compute_weights)"""
        return scores, None, attention_mask

    def compute_weights(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scores: Tensor,
        causal_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute linear attention with efficient operations.

        Args:
            q: Feature-mapped query tensor [B, H, L, D]
            k: Feature-mapped key tensor [B, H, L, D]
            v: Value tensor [B, H, L, D]
            scores: Placeholder (unused)
            causal_mask: Optional causal mask (unused)
            attention_mask: Optional padding mask

        Returns:
            Tuple of (attention_output, v)
        """
        batch_size, num_heads, seq_len, dim = q.size()
        v_dim = v.size(-1)

        # Apply padding mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(-1)
            k = k * mask
            v = v * mask

        # For non-causal attention: direct computation with O(LD²) complexity
        if not self.causal:
            # (1) Sum keys across sequence dimension
            k_sum = k.sum(dim=2)  # [B, H, D]

            # (2) Compute key-value outer products and sum
            kv = torch.einsum("bhld,bhle->bhde", k, v)  # [B, H, D, E]

            # (3) Compute outputs with linear complexity
            denom = torch.einsum("bhld,bhd->bhl", q, k_sum) + self.epsilon  # [B, H, L]
            numer = torch.einsum("bhld,bhde->bhle", q, kv)  # [B, H, L, E]

            # Normalize and return
            output = numer / denom.unsqueeze(-1)  # [B, H, L, E]
            return output, v

        # For causal attention: use cumulative sums
        else:
            # Compute causal linear attention using cumulative sums
            # For each position i, we need the sum of keys and key-values up to position i-1

            # First compute cumulative sums of keys and key-values
            # We shift these arrays to implement causality
            k_cumsum = torch.zeros_like(k)
            k_cumsum[:, :, 1:] = torch.cumsum(
                k[:, :, :-1], dim=2
            )  # Shift by 1 for causality

            # Compute key-value outer products
            kv = k.unsqueeze(-1) * v.unsqueeze(-2)  # [B, H, L, D, E]

            # Compute cumulative sum of key-values (with shift for causality)
            kv_cumsum = torch.zeros_like(kv)
            kv_cumsum[:, :, 1:] = torch.cumsum(kv[:, :, :-1], dim=2)

            # Compute outputs
            denom = torch.sum(q * k_cumsum, dim=3) + self.epsilon  # [B, H, L]
            numer = torch.sum(q.unsqueeze(-1) * kv_cumsum, dim=3)  # [B, H, L, E]

            # Normalize and return
            output = numer / denom.unsqueeze(-1)  # [B, H, L, E]
            return output, v

    def compute_outputs(self, weights: Tensor, v: Tensor) -> Tensor:
        """Apply normalization and dropout to outputs"""
        # Apply normalization for numerical stability
        batch_size, num_heads, seq_len, head_dim = weights.shape
        shaped_weights = weights.reshape(-1, head_dim)
        normalized = self.norm(shaped_weights)
        weights = normalized.view(batch_size, num_heads, seq_len, head_dim)

        return self.dropout(weights)


class Stickbreaking(ScaledDotProduct):
    """
    Implements Stickbreaking Attention mechanism.
    https://github.com/IBM/ModuleFormer
    """

    __version__ = "0.1.0"

    def __init__(self, config: Any) -> None:
        """
        Initialize stickbreaking attention module.

        Args:
            config: Configuration object containing attention parameters
        """
        # force no positional encoding
        setattr(config, "encoding", "nope")
        super().__init__(config)
        self.register_buffer("key_history", None)
        self.register_buffer("value_history", None)
        self.history_size: int = 32
        self.use_history: bool = True

    def compute_scores(
        self, q: Tensor, k: Tensor, v: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute attention scores with history management.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor

        Returns:
            Tuple of (query, key, value, attention scores)
        """
        if self.training and self.use_history:
            k, v = self._update_history(k, v)
        return super().compute_scores(q, k, v)

    def compute_weights(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scores: Tensor,
        causal_mask: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute stickbreaking attention weights from scores.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            scores: Attention scores
            causal_mask: Optional causal mask tensor

        Returns:
            Tuple of (attention weights, value tensor)
        """
        logits = scores
        batch_size, num_heads, seq_len, hist_len = logits.shape

        # Get cumulative weight matrix of appropriate size and expand it
        cum_weight = torch.tril(torch.ones(hist_len, hist_len, device=logits.device))

        # Compute stick-breaking weights
        z = torch.sigmoid(logits)
        log_beta = F.logsigmoid(-logits)
        if causal_mask is not None:
            z = z + causal_mask
            log_beta = log_beta + causal_mask

        # Compute cumulative log beta terms
        re_cum_log_beta = torch.einsum(
            "bhij,jk->bhik", log_beta, cum_weight.type_as(logits)
        )

        # Final attention weights
        weights = z * re_cum_log_beta.exp()

        return weights, v

    def _sample_kv_history(
        self, k_hist: Tensor, v_hist: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample fixed-size, aligned segments from key and value history tensors.

        Args:
            k_hist: Key history tensor
            v_hist: Value history tensor

        Returns:
            Tuple of (sampled key history, sampled value history)
        """
        _, _, seq_len, _ = k_hist.shape

        # If sequence length is less than or equal to desired history size,
        # return full history
        if seq_len <= self.history_size:
            return k_hist, v_hist

        # Generate random starting point that ensures we can get history_size tokens
        start_idx = torch.randint(0, seq_len - self.history_size + 1, (1,)).item()

        # Sample aligned segments from both tensors
        k_sample = k_hist[:, :, start_idx : start_idx + self.history_size, :]
        v_sample = v_hist[:, :, start_idx : start_idx + self.history_size, :]

        return k_sample, v_sample

    def _update_history(self, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Update and return history for keys and values.

        Args:
            k: Key tensor for current batch
            v: Value tensor for current batch

        Returns:
            Tuple of (updated key tensor, updated value tensor)
        """
        # First forward pass - initialize history
        if self.key_history is None or self.value_history is None:
            self.key_history = k.detach()
            self.value_history = v.detach()
            return k, v

        # Get current and history batch sizes
        curr_batch = k.size(0)
        hist_batch = self.key_history.size(0)

        # If current batch is smaller than history batch,
        # this means we have a longer sequence - return unmodified
        if curr_batch < hist_batch:
            return k, v

        # If current batch is larger than history batch,
        # this means we have shorter sequences - reset history
        if curr_batch > hist_batch:
            self.key_history = k.detach()
            self.value_history = v.detach()
            return k, v

        # Get aligned history samples
        hist_k, hist_v = self._sample_kv_history(self.key_history, self.value_history)

        # Concatenate [history slice, current sequence]
        new_k = torch.cat([hist_k, k], dim=2)
        new_v = torch.cat([hist_v, v], dim=2)

        # Update history
        self.key_history = new_k.detach()
        self.value_history = new_v.detach()

        return new_k, new_v

    # def _update_history(self, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
    #     """
    #     Update and return concatenated history for keys and values.
    #     Uses batch size as primary decision metric for history management,
    #     since larger batches correspond to smaller sequences that we want to concatenate.
    #     """
    #     # First forward pass - initialize history
    #     if self.key_history is None or self.value_history is None:
    #         self.key_history = k.detach()
    #         self.value_history = v.detach()
    #         return k, v

    #     # Get current and history batch sizes
    #     curr_batch = k.size(0)
    #     hist_batch = self.key_history.size(0)

    #     # If current batch is smaller than history batch,
    #     # this means we have a longer sequence - return unmodified
    #     if curr_batch < hist_batch:
    #         return k, v

    #     # If current batch is larger than history batch,
    #     # this means we have shorter sequences - reset history
    #     if curr_batch > hist_batch:
    #         self.key_history = k.detach()
    #         self.value_history = v.detach()
    #         return k, v

    #     # At this point batch sizes match, safe to concatenate
    #     try:
    #         new_k = torch.cat([self.key_history, k], dim=2)
    #         new_v = torch.cat([self.value_history, v], dim=2)
    #     except RuntimeError:
    #         # Safety fallback
    #         self.key_history = k.detach()
    #         self.value_history = v.detach()
    #         return k, v

    #     # Update history
    #     self.key_history = new_k.detach()
    #     self.value_history = new_v.detach()

    #     return new_k, new_v


class LinearQuery(nn.Linear):
    """
    Linear projection for query vectors.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize linear query projection.

        Args:
            *args: Arguments for nn.Linear
            **kwargs: Keyword arguments for nn.Linear
        """
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tuple[Tensor, float]:
        """
        Forward pass for query projection.

        Args:
            x: Input tensor

        Returns:
            Tuple of (projected tensor, auxiliary loss)
        """
        y = super().forward(x)
        aux_loss: float = 0
        return y, aux_loss


class LinearKeyValue(nn.Module):
    """
    Regular key/value projections.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, key_head_dim: int, value_head_dim: int
    ) -> None:
        """
        Initialize key and value projections.

        Args:
            hidden_size: Size of input dimension
            num_heads: Number of attention heads
            key_head_dim: Dimension of each key head
            value_head_dim: Dimension of each value head
        """
        super().__init__()
        self.key = nn.Linear(hidden_size, num_heads * key_head_dim, bias=False)
        self.value = nn.Linear(hidden_size, num_heads * value_head_dim, bias=False)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            + f"in_features={self.key.weight.size(0)}, "
            + f"out_keys={self.key.weight.size(1)}, "
            + f"out_values={self.value.weight.size(1)}"
            + ")"
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for key and value projections.

        Args:
            x: Input tensor

        Returns:
            Tuple of (key tensor, value tensor)
        """
        k = self.key(x)
        v = self.value(x)
        return k, v


class LowRankKeyValue(nn.Module):
    """
    A form of low-rank factorization for keys/values, inspired
    by "Tensor Product Attention Is All You Need":
    https://arxiv.org/abs/2501.06425
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        rank: int = 2,
    ) -> None:
        """
        Initialize low-rank key and value projections.

        Args:
            hidden_size: Size of input dimension
            num_heads: Number of attention heads
            key_head_dim: Dimension of each key head
            value_head_dim: Dimension of each value head
            rank: Rank of the factorization (default: 2)
        """
        super().__init__()
        self.num_heads: int = num_heads
        self.key_head_dim: int = key_head_dim
        self.value_head_dim: int = value_head_dim
        self.rank: int = rank

        # Define linear transformations for A projections
        self.key_a: nn.Linear = nn.Linear(
            hidden_size, self.num_heads * self.rank, bias=False
        )
        self.value_a: nn.Linear = nn.Linear(
            hidden_size, self.num_heads * self.rank, bias=False
        )

        # Define B projection parameters for K, V
        self.key_b: nn.Linear = nn.Linear(
            hidden_size, self.rank * self.key_head_dim, bias=False
        )
        self.value_b: nn.Linear = nn.Linear(
            hidden_size, self.rank * self.value_head_dim, bias=False
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            + f"in_features={self.key_a.weight.size(1)}, "
            + f"rank={self.rank}"
            + ")"
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for low-rank key and value projections.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Tuple of (key tensor, value tensor)
        """
        batch_size, seq_len, _ = x.size()

        # Compute intermediate variables A for K, and V
        A_k = self.key_a(x).view(batch_size, seq_len, self.num_heads, self.rank)
        A_v = self.value_a(x).view(batch_size, seq_len, self.num_heads, self.rank)

        # Compute intermediate variables B for K, and V
        B_k = self.key_b(x).view(batch_size, seq_len, self.rank, self.key_head_dim)
        B_v = self.value_b(x).view(batch_size, seq_len, self.rank, self.value_head_dim)

        # Reshape A_k, A_v
        A_k = A_k.view(batch_size * seq_len, self.num_heads, self.rank)
        A_v = A_v.view(batch_size * seq_len, self.num_heads, self.rank)

        # Reshape B_k, B_v
        B_k = B_k.view(batch_size * seq_len, self.rank, self.key_head_dim)
        B_v = B_v.view(batch_size * seq_len, self.rank, self.value_head_dim)

        k = (
            torch.bmm(A_k, B_k)
            .div_(self.rank)
            .view(batch_size, seq_len, self.num_heads, self.key_head_dim)
        )
        v = (
            torch.bmm(A_v, B_v)
            .div_(self.rank)
            .view(batch_size, seq_len, self.num_heads, self.value_head_dim)
        )

        return k, v


class MultiTokenAttention(nn.Module):
    """
    Implements Multi-Token Attention (MTA) with key-query convolution
    and head mixing convolution.
    https://arxiv.org/abs/2504.00927v1
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize Multi-Token Attention module.

        Args:
            config: Configuration object containing attention parameters
        """
        super().__init__()
        self.num_query_heads: int = config.num_heads * config.num_queries

        # Key-Query configuration
        self.query_kernel_size: int = 6
        self.key_kernel_size: int = 11

        # Head mixing configuration
        self.head_kernel_size: int = 2

        # Create a single grouped convolution for key-query convolution
        self.kq_conv: nn.Conv2d = nn.Conv2d(
            in_channels=self.num_query_heads,
            out_channels=self.num_query_heads,
            kernel_size=(self.query_kernel_size, self.key_kernel_size),
            padding="same",
            groups=self.num_query_heads,  # Each head gets its own filters
        )

        # Ensure we have divisible groups
        assert (
            self.num_query_heads % self.head_kernel_size == 0
        ), f"Number of heads ({self.num_query_heads}) must be divisible by head kernel size ({self.head_kernel_size})"

        self.num_head_groups: int = self.num_query_heads // self.head_kernel_size

        # For each group, create a weight matrix that mixes the heads
        self.head_mix_weights: nn.Parameter = nn.Parameter(
            torch.zeros(
                self.num_head_groups, self.head_kernel_size, self.head_kernel_size
            )
        )

        # In __init__:
        self.norm: nn.LayerNorm = nn.LayerNorm(config.head_size, eps=config.epsilon)

        # Initialize identity weights
        with torch.no_grad():
            for g in range(self.num_head_groups):
                for i in range(self.head_kernel_size):
                    self.head_mix_weights[g, i, i] = 1.0

        # Initialize as identity kernels
        self._init_identity_kernels()

    def _init_identity_kernels(self) -> None:
        """Initialize the kernels as identity (1 at center position)"""
        with torch.no_grad():
            # Initialize key-query convolution
            self.kq_conv.weight.zero_()
            center_q = self.query_kernel_size // 2
            center_k = self.key_kernel_size // 2

            # Set the center weight to 1.0 for each head's kernel
            for i in range(self.num_query_heads):
                self.kq_conv.weight[i, 0, center_q, center_k] = 1.0

    def key_query_convolution(self, scores: Tensor) -> Tensor:
        """
        Apply key-query convolution to attention scores with proper double masking
        as described in the paper's equation (4)

        Args:
            scores: Attention scores tensor of shape [batch_size, num_heads, seq_len, key_len]

        Returns:
            Processed attention scores
        """
        batch_size, num_heads, seq_len, key_len = scores.shape

        # Create causal masks
        # First mask (with 0s) - to be applied before convolution
        mask_before = torch.triu(
            torch.ones(seq_len, key_len, device=scores.device), diagonal=1
        )
        mask_before = (
            mask_before.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        )

        # Mask future tokens with 0 before convolution
        masked_scores = scores.masked_fill(mask_before.bool(), 0.0)

        # Apply convolution
        conv_scores = self.kq_conv(masked_scores)

        # Second mask (with -inf) - to be applied after convolution
        mask_after = mask_before.clone()

        # Mask future tokens with -inf after convolution
        final_scores = conv_scores.masked_fill(mask_after.bool(), -1e9)

        return final_scores

    def head_mixing_convolution(self, attention_weights: Tensor) -> Tensor:
        """
        Apply head mixing convolution to attention weights (post-softmax)
        using fully vectorized operations

        Args:
            attention_weights: Attention weights tensor of shape [batch_size, num_heads, seq_len, key_len]

        Returns:
            Mixed attention weights
        """
        batch_size, num_heads, seq_len, key_len = attention_weights.shape

        # Reshape to separate head groups: [batch_size, num_groups, head_kernel_size, seq_len, key_len]
        reshaped = attention_weights.view(
            batch_size, self.num_head_groups, self.head_kernel_size, seq_len, key_len
        )

        # Vectorized mixing using einsum:
        # - 'b' is batch dimension
        # - 'g' is group dimension
        # - 'i,j' are the source and target head indices within a group
        # - 's,k' are sequence and key dimensions
        # - 'w[g,i,j]' applies mixing weights for group g from source head j to target head i
        mixed_weights = torch.einsum(
            "bgisk,gij->bgjsk", reshaped, self.head_mix_weights
        )

        # Reshape back to original shape
        return mixed_weights.reshape(batch_size, num_heads, seq_len, key_len)

    def group_norm(self, attention_output: Tensor) -> Tensor:
        """
        Apply normalization with much less reshaping

        Args:
            attention_output: Attention output tensor of shape [batch_size, num_heads, seq_len, head_dim]

        Returns:
            Normalized attention output
        """
        batch_size, num_heads, seq_len, head_dim = attention_output.shape

        # We'll normalize each head vector independently
        # Reshape to [B*H*S, D] to apply LayerNorm efficiently
        reshaped = attention_output.reshape(-1, head_dim)

        # Apply norm to each head vector
        normalized = self.norm(reshaped)

        # Reshape back to original shape
        normalized = normalized.view(batch_size, num_heads, seq_len, head_dim)

        return normalized


class UniversalAttentionGate(nn.Module):
    """
    According to MEGA, "Single-head gated attention has been empirically
    shown [to be] as performant as vanilla multi-head attention."
    https://arxiv.org/abs/2209.10655
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize universal attention gate module.

        Args:
            config: Configuration object containing attention parameters
        """
        super().__init__()
        self.num_queries: int = config.num_queries
        self.hidden_size: int = config.hidden_size
        self.approximator: nn.Module = DENSE_REGISTRY.get("mlp")(
            config, activation=config.activation
        )

    def forward(self, inputs: Tensor, weights: Tensor) -> Tensor:
        """
        Forward pass to apply gating to attention outputs.

        Args:
            inputs: Original input tensor of shape [batch_size, seq_len, hidden_size]
            weights: Attention weights tensor of shape [batch_size, seq_len, num_queries*hidden_size]

        Returns:
            Gated attention output
        """
        batch_size, seq_len = inputs.shape[:2]

        if self.num_queries > 1:
            # Reshape weights to separate queries
            # From [B, S, Q*H] -> [B*Q, S, H]
            weights = (
                weights.view(batch_size, seq_len, self.num_queries, self.hidden_size)
                .transpose(1, 2)
                .reshape(batch_size * self.num_queries, seq_len, self.hidden_size)
            )

            # Repeat inputs for each query
            # From [B, S, H] -> [B*Q, S, H]
            inputs = (
                inputs.unsqueeze(1)
                .expand(-1, self.num_queries, -1, -1)
                .reshape(batch_size * self.num_queries, seq_len, self.hidden_size)
            )

            # Generate gates with original hidden size
            gates = self.approximator(inputs)  # [B*Q, S, H]

            # Apply gates and reshape back
            gated = gates * weights  # [B*Q, S, H]

            # Reshape back to original format
            # From [B*Q, S, H] -> [B, S, Q*H]
            return (
                gated.view(batch_size, self.num_queries, seq_len, self.hidden_size)
                .transpose(1, 2)
                .reshape(batch_size, seq_len, self.num_queries * self.hidden_size)
            )
        else:
            # Simple case: direct gating
            return self.approximator(inputs) * weights  # [B, S, H]


class PraxisGatedEMA(nn.Module):
    """
    Inspired by MEGA, this class implements a simple EMA into an attention mechanism,
    encouraging inductive biases in the model.
    Reference: https://arxiv.org/abs/2209.10655
    Original Code: https://github.com/facebookresearch/mega/blob/main/fairseq/modules/exponential_moving_average.py
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize gated EMA module.

        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        self.embed_dim: int = config.hidden_size
        self.ndim: int = 3  # Adjust as needed
        self.scale: float = math.sqrt(1.0 / self.ndim)

        # Truncation parameter to limit kernel size
        self.truncation: Optional[int] = None  # Set to a value like 256 if needed

        # EMA parameters
        self.delta: nn.Parameter = nn.Parameter(
            torch.Tensor(self.embed_dim, self.ndim, 1)
        )
        self.alpha: nn.Parameter = nn.Parameter(
            torch.Tensor(self.embed_dim, self.ndim, 1)
        )
        self.beta: nn.Parameter = nn.Parameter(
            torch.Tensor(self.embed_dim, self.ndim, 1)
        )
        self.gamma: nn.Parameter = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim))
        self.omega: nn.Parameter = nn.Parameter(torch.Tensor(self.embed_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters with appropriate distributions"""
        with torch.no_grad():
            nn.init.normal_(self.delta, mean=0.0, std=0.2)
            nn.init.normal_(self.alpha, mean=0.0, std=0.2)
            val = torch.ones(self.ndim, 1)
            if self.ndim > 1:
                idx = torch.tensor(list(range(1, self.ndim, 2)))
                val.index_fill_(0, idx, -1.0)
            self.beta.normal_(mean=0.0, std=0.02).add_(val)
            nn.init.normal_(self.gamma, mean=0.0, std=1.0)
            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the gated EMA module.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]

        Returns:
            Processed tensor after applying EMA
        """
        # Compute residual
        residual = x * self.omega  # Shape: (batch_size, seq_len, embed_dim)

        # Compute EMA
        ema_x = self._compute_ema(x)  # Shape: (batch_size, seq_len, embed_dim)

        # Combine EMA output with residual and apply activation function
        y = F.silu(ema_x + residual)  # Shape: (batch_size, seq_len, embed_dim)

        return y

    def _calc_coeffs(self) -> Tuple[Tensor, Tensor]:
        """
        Calculate EMA coefficients.

        Returns:
            Tuple of (p, q) coefficients
        """
        p = torch.sigmoid(self.delta)  # (embed_dim, ndim, 1)
        alpha = torch.sigmoid(self.alpha)  # (embed_dim, ndim, 1)
        q = 1.0 - p * alpha  # (embed_dim, ndim, 1)
        return p, q

    def _compute_kernel(self, seq_len: int) -> Tensor:
        """
        Compute the EMA kernel.

        Args:
            seq_len: Length of the sequence

        Returns:
            Computed kernel tensor
        """
        kernel_size = (
            seq_len if self.truncation is None else min(self.truncation, seq_len)
        )
        # Compute coefficients
        p, q = self._calc_coeffs()
        # Compute kernel
        t = torch.arange(kernel_size, device=p.device).view(1, 1, kernel_size)
        log_q = torch.log(q)
        vander = t * log_q  # (embed_dim, ndim, kernel_size)
        kernel = (p * self.beta) * torch.exp(vander)  # (embed_dim, ndim, kernel_size)
        kernel = torch.einsum(
            "dnl,dn->dl", kernel, self.gamma * self.scale
        )  # (embed_dim, kernel_size)
        return kernel

    def _compute_ema(self, x: Tensor) -> Tensor:
        """
        Compute EMA using FFT-based convolution.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]

        Returns:
            EMA output tensor of same shape
        """
        # x: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.size()
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)

        # Compute kernel
        kernel = self._compute_kernel(seq_len)  # (embed_dim, kernel_size)
        kernel_size = kernel.size(1)

        # Zero-pad kernel to match seq_len if necessary
        if kernel_size < seq_len:
            padding = seq_len - kernel_size
            kernel = F.pad(kernel, (0, padding))

        # Perform convolution using FFT
        fft_len = 2 * seq_len
        x_f = torch.fft.rfft(
            x.float(), n=fft_len, dim=2
        )  # (batch_size, embed_dim, fft_len//2+1)
        k_f = torch.fft.rfft(
            kernel.float(), n=fft_len, dim=1
        )  # (embed_dim, fft_len//2+1)

        # Multiply in frequency domain
        y_f = x_f * k_f.unsqueeze(0)  # Broadcasting over batch_size
        y = torch.fft.irfft(y_f, n=fft_len, dim=2)[
            ..., :seq_len
        ]  # (batch_size, embed_dim, seq_len)
        y = y.type_as(x)

        # Transpose back to (batch_size, seq_len, embed_dim)
        y = y.transpose(1, 2)
        return y


class VanillaMHA(nn.MultiheadAttention):
    """
    Standard multi-head attention implementation using PyTorch's nn.MultiheadAttention.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize vanilla multi-head attention module.

        Args:
            config: Configuration object containing attention parameters
        """
        super().__init__(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            bias=False,
            batch_first=True,
        )

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tensor, Optional[Tensor], int]:
        """
        Forward pass of vanilla multi-head attention.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask tensor
            past_key_values: Optional key-value cache (unused in this implementation)

        Returns:
            Tuple containing:
            - Output tensor after attention
            - None for the key-value cache (not used)
            - 0 for auxiliary loss (not used)
        """
        # scores shape: [B, S, E]
        seq_len = inputs.size(1)
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=inputs.device), diagonal=1
        ).bool()
        # Compute SDPA
        outputs, _ = super().forward(
            query=inputs,
            key=inputs,
            value=inputs,
            need_weights=False,
            is_causal=True,
            attn_mask=causal_mask,
        )
        layer_kv: Optional[Tensor] = None
        return outputs, layer_kv, 0


# Registry of available attention mechanisms
ATTENTION_REGISTRY: Dict[str, Type[nn.Module]] = {
    "standard": PraxisAttention,
    "vanilla": VanillaMHA,
    "pk": ProductKeyAttention,
}
