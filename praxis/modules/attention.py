import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from praxis.modules.encoding import ENCODING_REGISTRY
from praxis.modules.smear import PraxisSMEAR


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
        self.hidden_size = config.num_dims
        self.num_heads = config.num_heads
        self.num_queries = config.num_queries
        self.num_query_heads = self.num_heads * self.num_queries
        self.head_dim = self.hidden_size // self.num_heads

        # Set the core attention mechanism
        self.linear = config.linear
        self.differential = config.differential
        self.stickbreaking = config.stickbreaking
        assert (
            sum([self.differential, self.stickbreaking, self.linear]) <= 1
        ), "Only one of differential, stickbreaking, or linear attention can be used at a time."

        # Query and key projections for differential heads
        self.multiplier = 2 if self.differential else 1
        self.query = nn.Linear(
            self.hidden_size,
            self.num_query_heads * self.head_dim * self.multiplier,
            bias=False,
        )
        self.memory = config.memory
        if config.memory:

            class AttentionKey(nn.Module):
                """Simple wrapper for attention key projection"""

                def __init__(self, hidden_size, num_heads, head_dim):
                    super().__init__()
                    self.key = nn.Linear(
                        hidden_size,
                        num_heads * head_dim,
                        bias=False,
                    )

                def forward(self, x):
                    return self.key(x)

            # Create expert modules
            num_experts = 3
            experts = [
                AttentionKey(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim * self.multiplier,
                )
                for _ in range(num_experts)
            ]

            self.key = PraxisSMEAR(config, experts)

        else:
            self.key = nn.Linear(
                self.hidden_size,
                self.num_heads * self.head_dim * self.multiplier,
                bias=False,
            )
        self.value = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )

        # The core attention mechanism
        scale_encoding = True
        if self.stickbreaking:
            self.algorithm = Stickbreaking(config)
            config.encoding = "nope"
            scale_encoding = False
        elif self.differential:
            self.algorithm = Differential(config)
        elif self.linear:
            self.algorithm = LinearAttention(config)
        else:
            self.algorithm = ScaledDotProduct(config)

        # For handling length extrapolation
        self.encoding = ENCODING_REGISTRY[config.encoding](config, scale_encoding)

        # Standard output projection
        self.output = nn.Linear(
            self.num_query_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(self, inputs: Tensor, attention_mask: Tensor):
        batch_size, seq_len, _ = inputs.shape

        # Compute queries, keys, and values
        q = (
            self.query(inputs)
            .view(batch_size, -1, self.num_query_heads, self.head_dim * self.multiplier)
            .transpose(1, 2)
        )
        k = (
            self.key(inputs)
            .view(batch_size, -1, self.num_heads, self.head_dim * self.multiplier)
            .transpose(1, 2)
        )
        v = (
            self.value(inputs)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Expand k and v to match number of query heads
        # Shape: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_q_heads, seq_len, head_dim)
        if self.num_queries > 1:
            k = k.repeat_interleave(self.num_queries, dim=1)
            v = v.repeat_interleave(self.num_queries, dim=1)

        # Pre-scoring positional encoding
        q, k, v = self.encoding.before_scores(q, k, v)

        # Compute attention scores
        q, k, v, scores = self.algorithm.compute_scores(q, k, v)
        hist_len = k.size(2)

        # Post-scoring positional encoding
        scores = self.encoding.after_scores(scores)

        # Apply masks
        scores, causal_mask, attention_mask = self.algorithm.apply_masking(
            scores, attention_mask, seq_len, hist_len, self.causal
        )

        # Compute attention weights
        weights = self.algorithm.compute_weights(
            q, k, v, scores, causal_mask, attention_mask
        )

        # Reshape for output projection
        weights = weights.transpose(1, 2).reshape(
            batch_size, seq_len, -1
        )  # Shape: (batch_size, seq_len, num_heads * head_dim)

        # Output projection
        return self.output(weights)


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
        self.num_query_heads = self.num_heads * config.num_queries
        self.head_dim = self.hidden_size // self.num_heads
        # Force exploration of attention subnetworks
        self.dropout = nn.Dropout(config.dropout)

    def _compute_score(self, q, k):
        scaling = 1.0 / math.sqrt(self.head_dim)
        return torch.matmul(q, k.transpose(-2, -1)) * scaling

    def compute_scores(self, q, k, v):
        scores = [self._compute_score(q, k)]
        return q, k, v, scores

    def compute_weights(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scores,
        causal_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ):
        weights = [self.dropout(F.softmax(score, dim=-1)) for score in scores]
        return self._compute_outputs(weights[0], v)

    def _compute_outputs(self, weights, v):
        # Compute attention output
        return weights @ v  # Shape: (batch_size, num_heads, seq_len, head_dim)

    def apply_masking(self, scores, attention_mask, seq_len, hist_len, causal):
        causal_mask = None
        if causal:
            causal_mask = (
                torch.triu(
                    torch.full((seq_len, hist_len), -1e9, device=scores[0].device),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = [score + causal_mask for score in scores]

        # Expand attention mask to historical length
        attention_mask = F.pad(
            attention_mask, (hist_len - attention_mask.size(-1), 0), value=1
        )

        attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        scores = [score + attention_mask for score in scores]
        return scores, causal_mask, attention_mask


class LinearAttention(ScaledDotProduct):
    """
    Implements Linear Attention using kernel feature maps.
    Based on 'Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention'
    """

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        self.epsilon = 1e-6
        self.causal = config.causal

        # Feature map for positive definite kernel
        self.feature_map = lambda x: F.elu(x) + 1

        # Optional dropout
        self.dropout = nn.Dropout(config.dropout)

    def compute_scores(self, q: Tensor, k: Tensor, v: Tensor):
        """
        Instead of returning attention scores, we return the feature-mapped queries and keys.
        """
        # Apply the feature map
        q = self.feature_map(q)  # Shape: (B, H, L, D)
        k = self.feature_map(k)  # Shape: (B, H, L, D)

        return q, k, v, []  # Return an empty list for scores

    def apply_masking(self, scores, attention_mask, seq_len, hist_len, causal):
        return scores, None, attention_mask

    def compute_weights(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scores: List[Tensor],
        causal_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ):
        """
        Perform the linear attention computation here, using the feature-mapped q and k.
        """
        # Now, perform the linear attention computation
        B, H, L, D = v.size()

        # Apply mask to k and v if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # Shape: (B, 1, L, 1)
            k = k * mask
            v = v * mask

        if self.causal:
            # Implement causal linear attention using cumulative sums
            k_cumsum = torch.cumsum(k.transpose(2, 1), dim=2).transpose(
                2, 1
            )  # (B, H, L, D)
            kv_cumsum = torch.cumsum((k * v).transpose(2, 1), dim=2).transpose(
                2, 1
            )  # (B, H, L, D)

            # Compute denominator z
            z = torch.einsum("bhld,bhld->bhl", q, k_cumsum) + self.epsilon  # (B, H, L)

            # Apply attention mask to z
            if attention_mask is not None:
                z = z * attention_mask.unsqueeze(1)  # Shape: (B, 1, L)

            # Compute numerator
            output = torch.einsum("bhld,bhld->bhld", q, kv_cumsum)  # (B, H, L, D)
        else:
            # Non-causal linear attention
            k_sum = k.sum(dim=2)  # (B, H, D)
            kv_sum = torch.einsum("bhld,bhld->bhd", k, v)  # (B, H, D)

            # Compute denominator z
            z = torch.einsum("bhld,bhd->bhl", q, k_sum) + self.epsilon  # (B, H, L)

            # Apply attention mask to z
            if attention_mask is not None:
                z = z * attention_mask.unsqueeze(1)  # Shape: (B, 1, L)

            # Compute numerator
            output = torch.einsum("bhld,bhd->bhld", q, kv_sum)  # (B, H, L, D)

        # Normalize output
        output = output / z.unsqueeze(-1)  # (B, H, L, D)

        output = self.dropout(output)
        return output


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
            num_groups=self.num_query_heads,
            num_channels=self.num_query_heads * self.head_dim,
            eps=config.epsilon,
        )

    def compute_scores(self, q, k, v):
        # Split queries and keys
        q_chunks = q.chunk(q.size(-1) // self.head_dim, dim=-1)
        k_chunks = k.chunk(k.size(-1) // self.head_dim, dim=-1)
        # Compute differential attention scores
        scores = [
            self._compute_score(q_chunks[i], k_chunks[i]) for i in range(len(q_chunks))
        ]
        return q, k, v, scores

    def compute_weights(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scores,
        causal_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ):
        weights = [self.dropout(F.softmax(score, dim=-1)) for score in scores]
        # Compute scalar lambda
        lambda_scalar = (
            torch.exp(torch.dot(self.lambdas["q1"], self.lambdas["k1"]))
            - torch.exp(torch.dot(self.lambdas["q2"], self.lambdas["k2"]))
            + self.lambda_init
        )
        weights = weights[0] - lambda_scalar * weights[1]
        outputs = self._compute_outputs(weights, v)
        return self._normalize_outputs(outputs)

    def _normalize_outputs(self, weights):
        batch_size, num_heads, seq_len, head_dim = weights.shape
        # Reshape for GroupNorm
        attention_output = (
            weights.permute(0, 2, 1, 3)
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


class Stickbreaking(ScaledDotProduct):
    """
    Implements Stickbreaking Attention mechanism.
    https://github.com/IBM/ModuleFormer
    """

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        self.register_buffer("key_history", None)
        self.register_buffer("value_history", None)
        self.capacity = config.capacity
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
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        logits = scores[0]
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
        weights = self.dropout(z * re_cum_log_beta.exp())

        return self._compute_outputs(weights, v)

    def _get_random_slice(self, tensor: Tensor) -> Tensor:
        """Get random slice of history with size history_slice_size"""
        _, _, seq_len, _ = tensor.shape
        seg_len = int(seq_len * self.capacity)
        if seq_len <= seg_len:
            return tensor

        # Random starting point that ensures we can get history_slice_size tokens
        start_idx = torch.randint(0, seq_len - seg_len + 1, (1,)).item()
        return tensor[:, :, start_idx : start_idx + seg_len, :]

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

        try:
            # Get random slice from history
            hist_k = self._get_random_slice(self.key_history)
            hist_v = self._get_random_slice(self.value_history)

            # Concatenate [history slice, current sequence]
            new_k = torch.cat([hist_k, k], dim=2)
            new_v = torch.cat([hist_v, v], dim=2)

        except RuntimeError:
            # Safety fallback
            self.key_history = k.detach()
            self.value_history = v.detach()
            return k, v

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
