import math
from copy import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import DynamicCache

from praxis.functional import alpha_entmax, alpha_relu, ghostmax
from praxis.modules.dense import PraxisGLU, PraxisMLP
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

    def __init__(self, config: "AutoConfig"):
        super().__init__()
        self.causal = config.causal
        # Set the core attention mechanism
        self.linear = config.linear
        self.differential = config.differential
        self.stickbreaking = config.stickbreaking
        assert (
            sum([self.differential, self.stickbreaking, self.linear]) <= 1
        ), "Only one of differential, stickbreaking, or linear attention can be used at a time."

        hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_queries = config.num_queries
        self.num_query_heads = self.num_heads * self.num_queries
        self.kv_rank = config.kv_rank

        self.factor = 2 if self.differential else 1
        self.head_dim = hidden_size // self.num_heads // self.factor
        setattr(config, "head_size", self.head_dim)

        assert (
            sum([config.mega, config.gated]) <= 1
        ), "Only one of 'mega' or 'gated' can be used at a time."

        # For query gating
        self.ema = PraxisGatedEMA(config) if config.mega else False

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

        self.memory = config.memory
        self.chunk_size = 0
        if self.memory:
            self.chunk_size = 256
            self.memory = PraxisCompressiveMemory(config)

        # The core attention mechanism
        if self.stickbreaking:
            self.algorithm = Stickbreaking(config)
        elif self.differential:
            self.algorithm = Differential(config)
        # elif self.linear:
        #     self.algorithm = Linear(config)
        else:
            self.algorithm = ScaledDotProduct(config)

        # For handling length extrapolation
        self.encoding = ENCODING_REGISTRY[config.encoding](config)

        # For Multi-Token Attention
        self.mta = MultiTokenAttention(config) if config.mta else False

        # For attention gating
        self.gates = UniversalAttentionGate(config) if config.gated else False

        # Standard output projection
        self.output = nn.Linear(
            self.num_query_heads * self.head_dim, hidden_size, bias=False
        )

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Tensor = None,
        past_key_values: Tensor = None,
        block_ids: Tensor = None,
        current_depth: int = 0,
    ) -> Tensor:
        batch_size, seq_len, _ = inputs.shape
        aux_loss = 0

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

        outputs = []

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

            chunk_mask = (
                None if attention_mask is None else attention_mask[:, start_idx:end_idx]
            )
            chunk_block_ids = None
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
        attention_mask: Tensor,
        chunk_size: int,
        offset: int = 0,
        block_ids: Tensor = None,
    ) -> Tensor:
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

    def __init__(self, config: "AutoConfig"):
        super().__init__()
        self.meta = config.meta
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_query_heads = self.num_heads * config.num_queries
        self.head_dim = config.head_size
        # Force exploration of attention subnetworks
        self.dropout = nn.Dropout(config.dropout)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def compute_scores(self, q, k, v):
        scaling = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
        return q, k, v, scores

    def apply_masking(
        self, scores, attention_mask, block_ids, seq_len, hist_len, causal
    ):
        causal_mask = None
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

    def compute_weights(self, q, k, v, scores, *args, **kwargs):
        if "entmax" in self.meta:
            weights = alpha_entmax(scores, dim=-1)
        elif "relu" in self.meta:
            weights = alpha_relu(scores, dim=-1, alpha=1.5, tau=None)
        elif "softmax" in self.meta:
            weights = F.softmax(scores, dim=-1)
        else:
            weights = ghostmax(scores, dim=-1)
        return weights, v

    def compute_outputs(self, weights, v):
        return self.dropout(weights) @ v


class Differential(ScaledDotProduct):
    """
    This class implements Differential Attention, to filter the noise from attention maps:
    https://arxiv.org/abs/2410.05258
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig"):
        super().__init__(config)
        self.lambda_init = 0.8
        head_dim = config.head_size

        # Parameters for differential attention
        self.lambda_q1 = nn.Parameter(torch.zeros(head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(head_dim).normal_(mean=0, std=0.1))

        # GroupNorm should match the total channels
        self.norm = nn.GroupNorm(
            num_groups=self.num_query_heads,
            num_channels=self.num_query_heads * head_dim,
            eps=config.epsilon,
        )

    def compute_weights(self, q: Tensor, k: Tensor, v: Tensor, scores, *args, **kwargs):
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

    def compute_outputs(self, weights, v):
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


# class Linear(ScaledDotProduct):
#     """
#     Implements Linear Attention using kernel feature maps.
#     Based on 'Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention'
#     """

#     __version__ = "0.1.0"

#     def __init__(self, config: "AutoConfig"):
#         super().__init__(config)
#         self.epsilon = 1e-6
#         self.causal = config.causal

#         # Feature map for positive definite kernel
#         self.feature_map = lambda x: F.elu(x) + 1

#     def compute_scores(self, q: Tensor, k: Tensor, v: Tensor):
#         """
#         Instead of returning attention scores, we return the feature-mapped queries and keys.
#         """
#         # Apply the feature map
#         q = self.feature_map(q)  # Shape: (B, H, L, D)
#         k = self.feature_map(k)  # Shape: (B, H, L, D)

#         return q, k, v, []  # Return an empty list for scores

#     def apply_masking(self, scores, attention_mask, *args, **kwargs):
#         return scores, None, attention_mask

#     def compute_weights(
#         self,
#         q: Tensor,
#         k: Tensor,
#         v: Tensor,
#         scores: List[Tensor],
#         causal_mask: Optional[Tensor] = None,
#         attention_mask: Optional[Tensor] = None,
#     ):
#         """
#         Perform the linear attention computation here, using the feature-mapped q and k.
#         """
#         # Now, perform the linear attention computation
#         B, H, L, D = v.size()

#         # Apply mask to k and v if provided
#         if attention_mask is not None:
#             mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # Shape: (B, 1, L, 1)
#             k = k * mask
#             v = v * mask

#         if self.causal:
#             # Implement causal linear attention using cumulative sums
#             k_cumsum = torch.cumsum(k.transpose(2, 1), dim=2).transpose(
#                 2, 1
#             )  # (B, H, L, D)
#             kv_cumsum = torch.cumsum((k * v).transpose(2, 1), dim=2).transpose(
#                 2, 1
#             )  # (B, H, L, D)

#             # Compute denominator z
#             z = torch.einsum("bhld,bhld->bhl", q, k_cumsum) + self.epsilon  # (B, H, L)

#             # Apply attention mask to z
#             if attention_mask is not None:
#                 z = z * attention_mask.unsqueeze(1)  # Shape: (B, 1, L)

#             # Compute numerator
#             output = torch.einsum(
#                 "bhld,bhld->bhld", q, self.dropout(kv_cumsum)
#             )  # (B, H, L, D)
#         else:
#             # Non-causal linear attention
#             k_sum = k.sum(dim=2)  # (B, H, D)
#             kv_sum = torch.einsum("bhld,bhld->bhd", k, v)  # (B, H, D)

#             # Compute denominator z
#             z = torch.einsum("bhld,bhd->bhl", q, k_sum) + self.epsilon  # (B, H, L)

#             # Apply attention mask to z
#             if attention_mask is not None:
#                 z = z * attention_mask.unsqueeze(1)  # Shape: (B, 1, L)

#             # Compute numerator
#             output = torch.einsum(
#                 "bhld,bhd->bhld", q, self.dropout(kv_sum)
#             )  # (B, H, L, D)

#         # Normalize output
#         output = output / z.unsqueeze(-1)  # (B, H, L, D)

#         return output


class Stickbreaking(ScaledDotProduct):
    """
    Implements Stickbreaking Attention mechanism.
    https://github.com/IBM/ModuleFormer
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig"):
        # force no positional encoding
        setattr(config, "encoding", "nope")
        super().__init__(config)
        self.register_buffer("key_history", None)
        self.register_buffer("value_history", None)
        self.history_size = 32
        self.use_history = True

    def compute_scores(self, q, k, v):
        if self.training and self.use_history:
            k, v = self._update_history(k, v)
        return super().compute_scores(q, k, v)

    def compute_weights(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scores: List[Tensor],
        causal_mask: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        y = super().forward(x)
        aux_loss = 0
        return y, aux_loss


class LinearKeyValue(nn.Module):
    """
    Regular key/value projections.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, key_head_dim: int, value_head_dim: int
    ):
        super().__init__()
        self.key = nn.Linear(hidden_size, num_heads * key_head_dim, bias=False)
        self.value = nn.Linear(hidden_size, num_heads * value_head_dim, bias=False)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + f"in_features={self.key.weight.size(0)}, "
            + f"out_keys={self.key.weight.size(1)}, "
            + f"out_values={self.value.weight.size(0)}"
            + ")"
        )

    def forward(self, x):
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
    ):
        super().__init__()
        self.num_heads = num_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.rank = rank

        # Define linear transformations for A projections
        self.key_a = nn.Linear(hidden_size, self.num_heads * self.rank, bias=False)
        self.value_a = nn.Linear(hidden_size, self.num_heads * self.rank, bias=False)

        # Define B projection parameters for K, V
        self.key_b = nn.Linear(hidden_size, self.rank * self.key_head_dim, bias=False)
        self.value_b = nn.Linear(
            hidden_size, self.rank * self.value_head_dim, bias=False
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + f"in_features={self.key_a.weight.size(1)}, "
            + f"rank={self.rank}"
            + ")"
        )

    def forward(self, x):
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

    def __init__(self, config):
        super().__init__()
        self.num_query_heads = config.num_heads * config.num_queries

        # Key-Query configuration
        self.query_kernel_size = 6
        self.key_kernel_size = 11

        # Head mixing configuration
        self.head_kernel_size = 2

        # Create a single grouped convolution for key-query convolution
        self.kq_conv = nn.Conv2d(
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

        self.num_head_groups = self.num_query_heads // self.head_kernel_size

        # For each group, create a weight matrix that mixes the heads
        self.head_mix_weights = nn.Parameter(
            torch.zeros(
                self.num_head_groups, self.head_kernel_size, self.head_kernel_size
            )
        )

        # In __init__:
        self.norm = nn.LayerNorm(config.head_size, eps=config.epsilon)

        # Initialize identity weights
        with torch.no_grad():
            for g in range(self.num_head_groups):
                for i in range(self.head_kernel_size):
                    self.head_mix_weights[g, i, i] = 1.0

        # Initialize as identity kernels
        self._init_identity_kernels()

    def _init_identity_kernels(self):
        """Initialize the kernels as identity (1 at center position)"""
        with torch.no_grad():
            # Initialize key-query convolution
            self.kq_conv.weight.zero_()
            center_q = self.query_kernel_size // 2
            center_k = self.key_kernel_size // 2

            # Set the center weight to 1.0 for each head's kernel
            for i in range(self.num_query_heads):
                self.kq_conv.weight[i, 0, center_q, center_k] = 1.0

    # def key_query_convolution(self, scores):
    #     """
    #     Apply key-query convolution to attention scores
    #     """
    #     batch_size, num_heads, seq_len, key_len = scores.shape
    #     # Apply key-query convolution directly - no reshaping needed
    #     return self.kq_conv(scores)
    def key_query_convolution(self, scores):
        """
        Apply key-query convolution to attention scores with proper double masking
        as described in the paper's equation (4)
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

    def head_mixing_convolution(self, attention_weights):
        """
        Apply head mixing convolution to attention weights (post-softmax)
        using fully vectorized operations
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

    def group_norm(self, attention_output):
        """
        Apply normalization with much less reshaping
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

    def __init__(self, config: "AutoConfig"):
        super().__init__()
        self.num_queries = config.num_queries
        self.hidden_size = config.hidden_size
        self.approximator = PraxisMLP(config, activation=config.activation)

    def forward(self, inputs: Tensor, weights: Tensor) -> Tensor:
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

    def __init__(self, config: "AutoConfig"):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.ndim = 3  # Adjust as needed
        self.scale = math.sqrt(1.0 / self.ndim)

        # Truncation parameter to limit kernel size
        self.truncation = None  # Set to a value like 256 if needed

        # EMA parameters
        self.delta = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim, 1))
        self.alpha = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim, 1))
        self.beta = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim, 1))
        self.gamma = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim))
        self.omega = nn.Parameter(torch.Tensor(self.embed_dim))

        self.reset_parameters()

    def reset_parameters(self):
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
        # Compute residual
        residual = x * self.omega  # Shape: (batch_size, seq_len, embed_dim)

        # Compute EMA
        ema_x = self._compute_ema(x)  # Shape: (batch_size, seq_len, embed_dim)

        # Combine EMA output with residual and apply activation function
        y = F.silu(ema_x + residual)  # Shape: (batch_size, seq_len, embed_dim)

        return y

    def _calc_coeffs(self):
        p = torch.sigmoid(self.delta)  # (embed_dim, ndim, 1)
        alpha = torch.sigmoid(self.alpha)  # (embed_dim, ndim, 1)
        q = 1.0 - p * alpha  # (embed_dim, ndim, 1)
        return p, q

    def _compute_kernel(self, seq_len: int) -> Tensor:
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
    def __init__(self, config: "AutoConfig"):
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
        attention_mask: Tensor,
        past_key_values: Tensor = None,
        *args,
        **kwargs,
    ):
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
        layer_kv = None
        return outputs, layer_kv, 0


ATTENTION_REGISTRY = {
    "standard": PraxisAttention,
    "vanilla": VanillaMHA,
    "pk": ProductKeyAttention,
}
