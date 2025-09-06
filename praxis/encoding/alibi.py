import math
from typing import Optional, Tuple, TypeVar

import torch
import torch.nn.functional as F
from torch import nn

from praxis.encoding.nope import NoPE

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class ALiBi(NoPE):
    """
    This class implements Attention with Linear Biases (ALiBi), which is a form of
    length extrapolation that does not require trainable parameters.
    https://arxiv.org/abs/2108.12409
    """

    __version__ = "0.1.0"

    def __init__(self, config: ConfigType, *args, **kwargs):
        """
        Initialize ALiBi positional encoding.

        Args:
            config: Model configuration object
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(config)

    def compute_slopes(self, num_heads: int, device: torch.device) -> torch.Tensor:
        """
        Compute ALiBi slopes based on number of attention heads.

        Slopes decrease exponentially with head index, allowing different
        heads to focus on different position differences.

        Args:
            num_heads: Number of attention heads
            device: Device to create tensor on

        Returns:
            Tensor of slopes for each attention head
        """
        return 2 ** (-8 * torch.arange(1, num_heads + 1, device=device) / num_heads)

    def before_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        offset: int = 0,
        block_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pass through inputs unchanged (ALiBi applies biases after score computation).

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            offset: Position offset
            block_ids: Optional block IDs for segmented attention

        Returns:
            Unmodified (q, k, v) tensors
        """
        return q, k, v

    def after_scores(
        self,
        scores: torch.Tensor,
        offset: int = 0,
        block_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply ALiBi position-dependent biases to attention scores.

        Args:
            scores: Attention scores of shape [batch_size, num_heads, seq_len, seq_len]
            offset: Position offset for continuous positions
            block_ids: Optional block IDs for segmented attention

        Returns:
            Modified attention scores with ALiBi biases applied
        """
        batch_size, num_heads, query_len, key_len = scores.shape
        device = scores.device

        if block_ids is not None and block_ids.size(1) != 1:
            # Use vectorized position computation
            positions = self._compute_relative_positions_vectorized(block_ids, device)
            positions = positions.float()

            # Create attention mask for cross-sequence interactions
            seq_mask = block_ids.unsqueeze(-1) == block_ids.unsqueeze(-2)
            special_mask = (block_ids != -1).unsqueeze(-1) & (
                block_ids != -1
            ).unsqueeze(-2)
            valid_mask = seq_mask & special_mask

            # Compute masked position differences
            pos_diff = positions.unsqueeze(-1) - positions.unsqueeze(-2)
            pos_diff = pos_diff * valid_mask  # Zero out cross-sequence differences
        else:
            # Handle asymmetric attention (different query and key lengths)
            query_positions = torch.arange(
                query_len, dtype=torch.float32, device=device
            )
            key_positions = torch.arange(key_len, dtype=torch.float32, device=device)

            query_positions = query_positions.unsqueeze(0).expand(batch_size, query_len)
            key_positions = key_positions.unsqueeze(0).expand(batch_size, key_len)

            query_positions = query_positions + offset
            key_positions = key_positions + offset

            # Compute position differences: [batch, query_len, key_len]
            pos_diff = query_positions.unsqueeze(-1) - key_positions.unsqueeze(-2)

        # Apply ALiBi slopes
        slopes = self.compute_slopes(num_heads, device)
        biases = slopes.view(1, num_heads, 1, 1) * pos_diff.unsqueeze(1)

        return scores - biases

    def _compute_relative_positions_vectorized(
        self, block_ids: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Compute relative positions respecting block boundaries.

        Args:
            block_ids: Block IDs tensor of shape [batch_size, seq_len]
            device: Device to create tensors on

        Returns:
            Tensor of positions respecting block boundaries
        """
        # Same implementation as in RoPE
        mask = block_ids != -1
        positions = torch.cumsum(mask, dim=-1)
        boundaries = F.pad(block_ids[:, 1:] != block_ids[:, :-1], (1, 0), value=True)
        reset_mask = torch.cumsum(boundaries, dim=-1)
        segment_positions = (
            positions
            - positions.masked_fill(~mask, 0)
            .masked_fill(~boundaries, 0)
            .cummax(dim=-1)[0]
        )
        return segment_positions * mask
