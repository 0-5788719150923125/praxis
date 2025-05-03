import math
from typing import Optional, Tuple, TypeVar

import torch
import torch.nn.functional as F
from torch import nn

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class NoPE(nn.Module):
    """
    Implementation of NoPE (No Position Encoding) with head-wise attention scaling.
    https://arxiv.org/abs/2404.12224
    """

    __version__ = "0.1.0"

    def __init__(self, config: ConfigType):
        """
        Initialize NoPE with head-wise attention scaling.

        Args:
            config: Model configuration object containing attention settings
        """
        super().__init__()
        self.num_query_heads = config.num_heads * config.num_queries
        self.head_scales = nn.Parameter(torch.linspace(-1.2, 1.2, self.num_query_heads))

    def before_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        offset: int = 0,
        block_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply scaling to queries before computing attention scores.

        Args:
            q: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
            v: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
            offset: Position offset (unused in NoPE, for API compatibility)
            block_ids: Optional block IDs for segmented attention

        Returns:
            Tuple of (scaled_queries, keys, values)
        """
        # Get base scaling factor
        head_dim = q.size(-1)
        base_scale = 1.0 / math.sqrt(head_dim)

        # Reshape scales for broadcasting
        scaling = self.head_scales.view(1, -1, 1, 1) * base_scale

        # For Differential Attention
        if q.size(1) > self.head_scales.size(0):
            scaling = scaling.repeat_interleave(2, dim=1)

        # Apply scaling to queries
        return q * scaling, k, v

    def after_scores(
        self,
        scores: torch.Tensor,
        offset: int = 0,
        block_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process attention scores (no-op in NoPE).

        Args:
            scores: Attention scores tensor of shape [batch_size, num_heads, seq_len, seq_len]
            offset: Position offset (unused in NoPE, for API compatibility)
            block_ids: Optional block IDs for segmented attention

        Returns:
            Unmodified attention scores
        """
        return scores
