from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class MLAQuery(nn.Module):
    """
    Multi-head Latent Attention query projection with compression.
    Based on DeepSeek-V2: https://arxiv.org/abs/2405.04434

    Note: The DeepSeek-V2 implementation includes additional features not present here:
    - Two-stage projections with intermediate RMSNorm (q_a_proj -> norm -> q_b_proj)
    - Different terminology (uses "LoRA rank" instead of "compression dim")
    These features may be added in future versions if needed for performance parity.
    """

    def __init__(self, config) -> None:
        """
        Initialize MLA query projection.

        Args:
            config: Configuration object containing attention parameters
        """
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_heads
        self.num_queries: int = config.num_queries
        self.num_query_heads: int = self.num_heads * self.num_queries
        self.head_dim: int = config.head_size
        self.q_compression_dim: int = config.q_compression_dim
        self.rope_head_dim: int = config.rope_head_dim

        # Down-projection for query compression
        self.down_q: nn.Linear = nn.Linear(
            self.hidden_size, self.q_compression_dim, bias=False
        )

        # Up-projection for compressed queries
        self.up_q: nn.Linear = nn.Linear(
            self.q_compression_dim, self.num_query_heads * self.head_dim, bias=False
        )

        # Decoupled RoPE queries
        self.rope_q: nn.Linear = nn.Linear(
            self.q_compression_dim,
            self.num_query_heads * self.rope_head_dim,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, int]:
        """
        Forward pass for MLA query projection.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Tuple of (query tensor, auxiliary loss)
        """
        # Compress queries
        c_q = self.down_q(x)  # [B, S, q_compression_dim]

        # Up-project to full query dimension
        q_c = self.up_q(c_q)  # [B, S, num_query_heads * head_dim]

        # Generate RoPE queries (for position encoding)
        q_r = self.rope_q(c_q)  # [B, S, num_query_heads * rope_head_dim]

        # Concatenate compressed and RoPE queries
        batch_size, seq_len = x.shape[:2]

        # Reshape for concatenation
        q_c = q_c.view(batch_size, seq_len, self.num_query_heads, self.head_dim)
        q_r = q_r.view(batch_size, seq_len, self.num_query_heads, self.rope_head_dim)

        # Concatenate along feature dimension
        q = torch.cat([q_c, q_r], dim=3)  # [B, S, H, D + D_r]
        q = q.reshape(batch_size, seq_len, -1)  # [B, S, H * (D + D_r)]

        return q, 0


class MLAKeyValue(nn.Module):
    """
    Multi-head Latent Attention key-value compression.
    Based on DeepSeek-V2: https://arxiv.org/abs/2405.04434

    Note: The DeepSeek-V2 implementation includes additional features not present here:
    - Two-stage projections with intermediate RMSNorm (kv_a_proj -> norm -> kv_b_proj)
    - Explicit Multi-Query Attention (MQA) support in projections
    - More flexible RoPE scaling strategies (linear, dynamic, yarn)
    These features may be added in future versions if needed for performance parity.
    """

    def __init__(self, config) -> None:
        """
        Initialize MLA key-value compression.

        Args:
            config: Configuration object containing attention parameters
        """
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_heads
        self.head_dim: int = config.head_size
        self.kv_compression_dim: int = config.kv_compression_dim
        self.rope_head_dim: int = config.rope_head_dim

        # Joint compression for keys and values
        self.down_kv: nn.Linear = nn.Linear(
            self.hidden_size, self.kv_compression_dim, bias=False
        )

        # Up-projections for keys and values
        self.up_k: nn.Linear = nn.Linear(
            self.kv_compression_dim, self.num_heads * self.head_dim, bias=False
        )

        self.up_v: nn.Linear = nn.Linear(
            self.kv_compression_dim, self.num_heads * self.head_dim, bias=False
        )

        # Decoupled RoPE key (shared across heads)
        self.rope_k: nn.Linear = nn.Linear(
            self.hidden_size, self.rope_head_dim, bias=False
        )

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for MLA key-value compression.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Tuple of (key tensor, value tensor)
        """
        # Joint compression for keys and values
        c_kv = self.down_kv(x[0])  # [B, S, kv_compression_dim]

        # Up-project to get keys and values
        k_c = self.up_k(c_kv)  # [B, S, num_heads * head_dim]
        v = self.up_v(c_kv)  # [B, S, num_heads * head_dim]

        # Generate shared RoPE key
        k_r = self.rope_k(x[0])  # [B, S, rope_head_dim]

        # Combine compressed keys with RoPE keys
        batch_size, seq_len = x[0].shape[:2]

        # Reshape compressed keys
        k_c = k_c.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Expand RoPE key to all heads
        k_r = k_r.unsqueeze(2).expand(
            -1, -1, self.num_heads, -1
        )  # [B, S, H, rope_head_dim]

        # Concatenate compressed and RoPE keys
        k = torch.cat([k_c, k_r], dim=3)  # [B, S, H, D + D_r]
        k = k.reshape(batch_size, seq_len, -1)  # [B, S, H * (D + D_r)]

        return k, v
