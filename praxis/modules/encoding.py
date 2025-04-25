import math
from typing import Optional, Tuple, TypeVar

import torch
import torch.nn.functional as F
from torch import nn

ConfigType = TypeVar('ConfigType', bound='AutoConfig')


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
        block_ids: Optional[torch.Tensor] = None
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
        block_ids: Optional[torch.Tensor] = None
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
        block_ids: Optional[torch.Tensor] = None
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
        block_ids: Optional[torch.Tensor] = None
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
        batch_size, num_heads, seq_len, _ = scores.shape
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
            # Original continuous position handling
            positions = torch.arange(seq_len, dtype=torch.float32, device=device)
            positions = positions.unsqueeze(0).expand(batch_size, seq_len)
            positions = positions + offset
            pos_diff = positions.unsqueeze(-1) - positions.unsqueeze(-2)

        # Apply ALiBi slopes
        slopes = self.compute_slopes(num_heads, device)
        biases = slopes.view(1, num_heads, 1, 1) * pos_diff.unsqueeze(1)

        return scores - biases

    def _compute_relative_positions_vectorized(
        self, 
        block_ids: torch.Tensor, 
        device: torch.device
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


class RoPE(NoPE):
    """
    An implementation of Rotary Position Embeddings (RoPE).
    Supports Grouped Query Attention and odd head dimensions.
    """

    __version__ = "0.2.0"

    def __init__(self, config: ConfigType, *args, **kwargs):
        """
        Initialize Rotary Position Embeddings.
        
        Args:
            config: Model configuration object
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(config)
        self.inv_freq: Optional[torch.Tensor] = None
        self.theta: float = 10000.0
        self._cached_cos: Optional[torch.Tensor] = None
        self._cached_sin: Optional[torch.Tensor] = None
        self._cached_seq_length: Optional[int] = None

    def before_scores(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        offset: int = 0, 
        block_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to queries and keys.
        
        Args:
            q: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
            v: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
            offset: Position offset for continuous positions
            block_ids: Optional block IDs for segmented attention
            
        Returns:
            Tuple of (rotated_queries, rotated_keys, values)
        """
        q_seq_len = q.size(2)
        k_seq_len = k.size(2)
        device = q.device
        dtype = q.dtype
        head_dim = q.size(-1)

        # Compute embeddings using the longer sequence length
        max_seq_len = max(q_seq_len, k_seq_len)
        self._compute_rope_embeddings(
            head_dim, max_seq_len, device, dtype, offset, block_ids
        )

        # When using caching during inference
        if q_seq_len == 1 and k_seq_len > 1:
            # For queries: take the last position
            q_cos = self._cached_cos[:, :, -1:, :]
            q_sin = self._cached_sin[:, :, -1:, :]
        else:
            # During training: normal behavior
            q_cos = self._cached_cos[:, :, :q_seq_len, :]
            q_sin = self._cached_sin[:, :, :q_seq_len, :]

        k_cos = self._cached_cos[:, :, :k_seq_len, :]
        k_sin = self._cached_sin[:, :, :k_seq_len, :]

        # Apply rotations with different positional encodings
        q_rope = self._apply_rotary_pos_emb(q, q_cos, q_sin)
        k_rope = self._apply_rotary_pos_emb(k, k_cos, k_sin)

        return q_rope, k_rope, v

    def after_scores(
        self, 
        scores: torch.Tensor, 
        offset: int = 0, 
        block_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process attention scores (no-op in RoPE).
        
        Args:
            scores: Attention scores tensor of shape [batch_size, num_heads, seq_len, seq_len]
            offset: Position offset (unused in RoPE, for API compatibility)
            block_ids: Optional block IDs for segmented attention
            
        Returns:
            Unmodified attention scores
        """
        return scores

    def _compute_rope_embeddings(
        self, 
        head_dim: int, 
        seq_len: int, 
        device: torch.device, 
        dtype: torch.dtype, 
        offset: int = 0, 
        block_ids: Optional[torch.Tensor] = None
    ) -> None:
        """
        Compute rotary positional embeddings for the given parameters.
        
        Args:
            head_dim: Dimension of each attention head
            seq_len: Maximum sequence length to compute embeddings for
            device: Device to create tensors on
            dtype: Data type for the embeddings
            offset: Position offset for continuous positions
            block_ids: Optional block IDs for segmented attention
        """
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.theta
                ** (
                    2 * torch.arange(0, head_dim // 2, device=device).float() / head_dim
                )
            )

        if block_ids is not None and block_ids.size(1) != 1:
            positions = self._compute_relative_positions_vectorized(
                block_ids, device
            )  # Shape: [batch_size, seq_len]
        else:
            positions = (torch.arange(seq_len, device=device) + offset).unsqueeze(0)

        # Use batch dimension in einsum
        freqs = torch.einsum("bi,j->bij", positions, self.inv_freq)

        # Reshape for proper broadcasting
        cos = torch.cos(freqs)  # [batch_size, seq_len, head_dim//2]
        sin = torch.sin(freqs)  # [batch_size, seq_len, head_dim//2]

        # Stack and reshape to match original dimensions
        cos = torch.stack([cos, cos], dim=-1).view(cos.size(0), 1, cos.size(1), -1)
        sin = torch.stack([-sin, sin], dim=-1).view(sin.size(0), 1, sin.size(1), -1)

        self._cached_cos = cos.to(dtype)
        self._cached_sin = sin.to(dtype)
        self._cached_seq_length = seq_len

    def _compute_relative_positions_vectorized(
        self, 
        block_ids: torch.Tensor, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute relative positions respecting block boundaries.
        
        Args:
            block_ids: Block IDs tensor of shape [batch_size, seq_len]
            device: Device to create tensors on
            
        Returns:
            Tensor of positions respecting block boundaries
        """
        # Create mask for valid tokens
        mask = block_ids != -1

        # Create position indices
        positions = torch.cumsum(mask, dim=-1)

        # Create segment boundaries
        boundaries = torch.nn.functional.pad(
            block_ids[:, 1:] != block_ids[:, :-1], (1, 0), value=True
        )

        # Reset cumsum at boundaries
        reset_mask = torch.cumsum(boundaries, dim=-1)
        segment_positions = (
            positions
            - positions.masked_fill(~mask, 0)
            .masked_fill(~boundaries, 0)
            .cummax(dim=-1)[0]
        )

        # Zero out special token positions
        return segment_positions * mask

    def _apply_rotary_pos_emb(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply rotary position embeddings with proper handling of odd dimensions.
        
        Args:
            x: Input tensor to apply rotations to
            cos: Cosine part of the rotation
            sin: Sine part of the rotation
            
        Returns:
            Tensor with rotary positional embeddings applied
        """
        seq_len = x.size(2)

        # Ensure proper broadcasting
        cos = cos[:, :, :seq_len, :]
        sin = sin[:, :, :seq_len, :]

        # Split input into pairs (handles odd dimensions)
        x1, x2 = x.chunk(2, dim=-1)
        d1, d2 = x1.size(-1), x2.size(-1)

        # Pad x2 if head_dim is odd
        if d1 > d2:
            x2 = F.pad(x2, (0, 1))  # Pad last dimension with zero

        # Apply rotations using d1 consistently
        out1 = x1 * cos[..., :d1] - x2 * sin[..., :d1]
        out2 = x1 * sin[..., :d1] + x2 * cos[..., :d1]

        # Truncate out2 if head_dim is odd
        if d1 > d2:
            out2 = out2[..., :d2]

        return torch.cat([out1, out2], dim=-1)


ENCODING_REGISTRY = {"alibi": ALiBi, "nope": NoPE, "rope": RoPE}
