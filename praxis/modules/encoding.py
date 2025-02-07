import math

import torch
import torch.nn.functional as F
from torch import nn


class NoPE(nn.Module):
    """
    Implementation of NoPE (No Position Encoding) with head-wise attention scaling.
    https://arxiv.org/abs/2404.12224
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig"):
        super().__init__()
        self.num_query_heads = config.num_heads * config.num_queries
        self.head_scales = nn.Parameter(torch.linspace(-1.2, 1.2, self.num_query_heads))

    def before_scores(self, q, k, v, offset: int = 0, block_ids=None):
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

    def after_scores(self, scores, offset: int = 0, block_ids=None):
        return scores


class ALiBi(NoPE):
    """
    This class implements Attention with Linear Biases (ALiBi), which is a form of
    length extrapolation that does not require trainable parameters.
    https://arxiv.org/abs/2108.12409
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig", *args, **kwargs):
        super().__init__(config)

    def compute_slopes(self, num_heads: int, device: torch.device) -> torch.Tensor:
        """Compute ALiBi slopes based on number of attention heads."""
        return 2 ** (-8 * torch.arange(1, num_heads + 1, device=device) / num_heads)

    def before_scores(self, q, k, v, offset: int = 0, block_ids=None):
        return q, k, v

    def after_scores(self, scores, offset: int = 0, block_ids=None):
        batch_size, num_heads, seq_len, _ = scores.shape
        device = scores.device

        if block_ids is not None:
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

    def _compute_relative_positions_vectorized(self, block_ids, device):
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

    def __init__(self, config: "AutoConfig", *args, **kwargs):
        super().__init__(config)
        self.inv_freq = None
        self.theta = 10000.0
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = None

    def before_scores(self, q, k, v, offset: int = 0, block_ids=None):
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

    def after_scores(self, scores, offset: int = 0, block_ids=None):
        return scores

    def _compute_rope_embeddings(
        self, head_dim, seq_len, device, dtype, offset: int = 0, block_ids=None
    ):
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.theta
                ** (
                    2 * torch.arange(0, head_dim // 2, device=device).float() / head_dim
                )
            )

        if block_ids is not None:
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

    def _compute_relative_positions_vectorized(self, block_ids, device):
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

    def _apply_rotary_pos_emb(self, x, cos, sin):
        """Apply rotary position embeddings with proper handling of odd dimensions."""
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
