import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.functional import alpha_entmax, alpha_relu, ghostmax


class ScaledDotProduct(nn.Module):
    """
    This class implements scaled dot-product attention:
    https://paperswithcode.com/method/scaled
    """

    __version__ = "0.1.0"

    def __init__(self, config) -> None:
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
        scaling = 1.0 / math.sqrt(q.size(-1))
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

    def __init__(self, config) -> None:
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

    def __init__(self, config) -> None:
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

    def __init__(self, config) -> None:
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