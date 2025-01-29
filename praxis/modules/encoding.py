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

    def before_scores(self, q, k, v, offset: int = 0):
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

    def after_scores(self, scores, offset: int = 0):
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

    def before_scores(self, q, k, v, offset: int = 0):
        return q, k, v

    def after_scores(self, scores, offset: int = 0):
        batch_size, num_heads, seq_len, _ = scores.shape
        device = scores.device

        # Compute positions dynamically
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        # Add offset to positions
        positions = positions + offset
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)

        # Compute slopes dynamically based on actual number of heads
        slopes = self.compute_slopes(num_heads, device)
        biases = slopes.view(1, num_heads, 1, 1) * pos_diff.unsqueeze(1)

        return scores - biases


class RoPE(NoPE):
    """
    An upgraded implementation of Rotary Position Embeddings (RoPE).
    Maintains backward compatibility and supports odd head dimensions.
    """

    __version__ = "0.2.0"

    def __init__(self, config: "AutoConfig", *args, **kwargs):
        super().__init__(config)
        self.inv_freq = None
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = None

    def before_scores(self, q, k, v, offset: int = 0):
        seq_len = q.size(2)
        device = q.device
        dtype = q.dtype
        head_dim = q.size(-1)

        self._compute_rope_embeddings(head_dim, seq_len, device, dtype, offset)

        # Apply rotations with proper broadcasting
        q_rope = self._apply_rotary_pos_emb(q, self._cached_cos, self._cached_sin)
        k_rope = self._apply_rotary_pos_emb(k, self._cached_cos, self._cached_sin)

        return q_rope, k_rope, v

    def after_scores(self, scores, offset: int = 0):
        return scores

    def _compute_rope_embeddings(
        self, head_dim, seq_len, device, dtype, offset: int = 0
    ):
        """Compute sin and cos embeddings with offset."""
        if self._needs_recompute(seq_len, device, dtype, offset):
            # Lazy initialize frequency tensor
            if self.inv_freq is None:
                theta = 10000.0
                self.inv_freq = 1.0 / (
                    theta
                    ** (
                        2
                        * torch.arange(0, head_dim // 2, device=device).float()
                        / head_dim
                    )
                )

            # Create position indices with offset
            positions = torch.arange(seq_len, device=device) + offset

            # Compute outer product of positions and frequencies
            freqs = torch.einsum("i,j->ij", positions, self.inv_freq)

            # Create rotation matrices
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)

            # Stack properly to match alternating pairs
            cos = torch.stack([cos, cos], dim=-1).view(1, 1, seq_len, -1)
            sin = torch.stack([-sin, sin], dim=-1).view(1, 1, seq_len, -1)

            # Update cache
            self._cached_cos = cos.to(dtype)
            self._cached_sin = sin.to(dtype)
            self._cached_seq_length = seq_len

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

    def _needs_recompute(self, seq_len, device, dtype, offset):
        return (
            self._cached_seq_length is None
            or seq_len + offset > self._cached_seq_length
            or self._cached_cos.device != device
            or self._cached_cos.dtype != dtype
        )


ENCODING_REGISTRY = {"alibi": ALiBi, "nope": NoPE, "rope": RoPE}
