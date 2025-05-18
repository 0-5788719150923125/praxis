import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.dense import DENSE_REGISTRY


class MultiTokenAttention(nn.Module):
    """
    Implements Multi-Token Attention (MTA) with key-query convolution
    and head mixing convolution.
    https://arxiv.org/abs/2504.00927v1
    """

    def __init__(self, config) -> None:
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

    def __init__(self, config) -> None:
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


class GatedEMA(nn.Module):
    """
    Inspired by MEGA, this class implements a simple EMA into an attention mechanism,
    encouraging inductive biases in the model.
    Reference: https://arxiv.org/abs/2209.10655
    Original Code: https://github.com/facebookresearch/mega/blob/main/fairseq/modules/exponential_moving_average.py
    """

    def __init__(self, config) -> None:
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

    def __init__(self, config) -> None:
        """
        Initialize vanilla multi-head attention module.

        Args:
            config: Configuration object containing attention parameters
        """
        while config.hidden_size % config.num_heads != 0:
            setattr(config, "num_heads", config.num_heads - 1)
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
        attention_mask: Optional[Tensor] = None,
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
