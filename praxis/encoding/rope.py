import math
from typing import Optional, Tuple, TypeVar

import torch
import torch.nn.functional as F
from torch import nn

from praxis.encoding.nope import NoPE

ConfigType = TypeVar("ConfigType", bound="AutoConfig")

# Learned theta is bounded to this range via a sigmoid, so it can never
# explode or collapse. THETA_MIN sits in the 50-200 band Huang & Chen (IEEE
# Access 2025) found best. We init at THETA_INIT rather than the legacy
# 10000: the same study found 400-800 optimal at small scale, and tuning
# theta below 10000 is the whole point of this work.
THETA_MIN: float = 100.0
THETA_MAX: float = 1_000_000.0
THETA_INIT: float = 500.0


class RoPE(NoPE):
    """
    An implementation of Rotary Position Embeddings (RoPE).
    Supports Grouped Query Attention and odd head dimensions.

    The base ``theta`` is learned rather than fixed at 10000, with a per-depth
    log-theta delta so each recurrent-depth pass gets its own positional zoom
    (coarse-to-fine across depths). Parameterized in a bounded log space:
    theta = THETA_MIN * (THETA_MAX/THETA_MIN)^sigmoid(z_base + z_delta[depth]).
    z_base inits theta to THETA_INIT; z_delta is zero-init so every depth
    starts there and specializes from data.
    """

    def __init__(self, config: ConfigType, *args, **kwargs):
        """
        Initialize Rotary Position Embeddings.

        Args:
            config: Model configuration object
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(config)
        self.depth = config.depth
        # Bounded log-theta: shared base + zero-init per-depth delta. z_base is
        # the sigmoid logit that lands theta on THETA_INIT.
        s = math.log(THETA_INIT / THETA_MIN) / math.log(THETA_MAX / THETA_MIN)
        z_init = math.log(s / (1.0 - s))
        self.log_theta_base = nn.Parameter(torch.full((1,), z_init))
        self.depth_log_theta = nn.Embedding(self.depth, 1)
        nn.init.zeros_(self.depth_log_theta.weight)
        self._cached_cos: Optional[torch.Tensor] = None
        self._cached_sin: Optional[torch.Tensor] = None
        self._cached_seq_length: Optional[int] = None

    @property
    def theta(self) -> float:
        """Effective base theta (depth 0), for logging / inspection."""
        with torch.no_grad():
            return self._effective_theta(0).item()

    def _effective_theta(self, current_depth: int) -> torch.Tensor:
        """Bounded, learned theta for a given recurrent depth."""
        idx = torch.tensor(current_depth, device=self.log_theta_base.device)
        z = self.log_theta_base + self.depth_log_theta(idx).squeeze(-1)
        # span**sigmoid(z) written as exp(sigmoid(z)*log(span)): a scalar-base
        # power's backward needs log(base), which inductor can't lower when
        # dynamic=True makes the scalar a symbolic float (NYI log symbolic float).
        log_span = math.log(THETA_MAX / THETA_MIN)
        return THETA_MIN * torch.exp(log_span * torch.sigmoid(z))

    def _compute_inv_freq(
        self, head_dim: int, device: torch.device, current_depth: int
    ) -> torch.Tensor:
        """Inverse frequencies for every RoPE band (subclasses may subset)."""
        theta = self._effective_theta(current_depth).to(device)
        exponent = 2 * torch.arange(0, head_dim // 2, device=device).float() / head_dim
        return 1.0 / (theta**exponent)

    def _position_freqs(
        self,
        positions: torch.Tensor,
        inv_freq: torch.Tensor,
        current_depth: int,
    ) -> torch.Tensor:
        """Rotation angle per (position, band): linear in position for RoPE.

        Seam for subclasses (ArcHoPE) that warp the phase. Returns a tensor
        of shape [batch, seq_len, num_bands].
        """
        return torch.einsum("bi,j->bij", positions.float(), inv_freq)

    def before_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        offset: int = 0,
        block_ids: Optional[torch.Tensor] = None,
        current_depth: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to queries and keys.

        Args:
            q: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
            v: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
            offset: Position offset for continuous positions
            block_ids: Optional block IDs for segmented attention
            current_depth: Recurrent depth pass, selects the per-depth theta

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
            head_dim, max_seq_len, device, dtype, offset, block_ids, current_depth
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
        block_ids: Optional[torch.Tensor] = None,
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
        block_ids: Optional[torch.Tensor] = None,
        current_depth: int = 0,
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
            current_depth: Recurrent depth pass, selects the per-depth theta
        """
        # Recomputed each call: theta is learned, so inv_freq carries gradient.
        inv_freq = self._compute_inv_freq(head_dim, device, current_depth)

        if block_ids is not None and block_ids.size(1) != 1:
            positions = self._compute_relative_positions_vectorized(
                block_ids, device
            )  # Shape: [batch_size, seq_len]
        else:
            positions = (torch.arange(seq_len, device=device) + offset).unsqueeze(0)

        freqs = self._position_freqs(positions, inv_freq, current_depth)

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
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
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
