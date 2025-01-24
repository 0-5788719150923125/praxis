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
        # Pre-compute the ALiBi slopes
        slopes = 2 ** (-8 * torch.arange(1, config.num_heads + 1) / config.num_heads)
        self.register_buffer("slopes", slopes)
        self.register_buffer(
            "positions", torch.arange(config.context_length, dtype=torch.float32)
        )

    def before_scores(self, q, k, v, offset: int = 0):
        return q, k, v

    def compute_after(self, scores, offset: int = 0):
        batch_size, num_heads, seq_len, _ = scores[0].shape
        # Use offset positions
        positions = self.positions[:seq_len].unsqueeze(0).expand(batch_size, seq_len)
        # Add offset to positions
        positions = positions + offset
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        biases = self.slopes.view(1, num_heads, 1, 1) * pos_diff.unsqueeze(1)
        scores = [score - biases for score in scores]
        return scores


class RoPE(NoPE):
    """
    Implementation of Rotary Position Embeddings (RoPE).
    https://arxiv.org/abs/2104.09864
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig", *args, **kwargs):
        super().__init__(config)
        self.inv_freq = None
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = None

    # before_scores and after_scores remain the same
    def before_scores(self, q, k, v, offset: int = 0):
        seq_len = q.size(2)
        device = q.device
        dtype = q.dtype

        head_dim = q.size(-1)
        self._compute_rope_embeddings(head_dim, seq_len, device, dtype, offset)

        cos = self._cached_cos[:, :, :seq_len, :]
        sin = self._cached_sin[:, :, :seq_len, :]

        q_rope = self._apply_rotary_pos_emb(q, cos, sin)
        k_rope = self._apply_rotary_pos_emb(k, cos, sin)

        return q_rope, k_rope, v

    def after_scores(self, scores, offset: int = 0):
        return scores

    def _compute_rope_embeddings(
        self, head_dim, seq_len, device, dtype, offset: int = 0
    ):
        """Compute sin and cos embeddings with offset."""
        if (
            self._cached_seq_length is None
            or seq_len > self._cached_seq_length
            or self._cached_cos is None
            or self._cached_cos.device != device
            or self._cached_cos.dtype != dtype
        ):
            positions = torch.arange(seq_len, device=device) + offset

            if self.inv_freq is None:
                theta = 10000
                self.inv_freq = 1.0 / (
                    theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
                ).to(device)

            pos_emb = positions.unsqueeze(1) * self.inv_freq.unsqueeze(0)

            # Compute basic sin and cos
            cos = torch.cos(pos_emb)
            sin = torch.sin(pos_emb)

            # Stack properly to match Meta's implementation
            # For each position and frequency, create alternating pairs
            cos = torch.stack([cos, cos], dim=-1).view(1, 1, seq_len, -1)
            sin = torch.stack([-sin, sin], dim=-1).view(
                1, 1, seq_len, -1
            )  # Note the negative sign

            self._cached_cos = cos.to(dtype)
            self._cached_sin = sin.to(dtype)
            self._cached_seq_length = seq_len

    def _apply_rotary_pos_emb(self, x, cos, sin):
        """Apply rotary position embeddings with proper handling of odd dimensions."""
        # Split input into pairs (larger chunk first for odd dimensions)
        x1, x2 = x.chunk(2, dim=-1)
        d1, d2 = x1.size(-1), x2.size(-1)

        cos_0 = cos[:, :, :, 0::2]
        sin_0 = sin[:, :, :, 0::2]
        cos_1 = cos[:, :, :, 1::2]
        sin_1 = sin[:, :, :, 1::2]

        # For odd dimensions
        if d1 > d2:
            x2 = torch.nn.functional.pad(x2, (0, 1))
            if cos_1.size(-1) < d1:
                # Calculate exact padding needed
                pad_size = d1 - cos_1.size(-1)
                cos_1 = torch.nn.functional.pad(cos_1, (0, pad_size), value=0.0)
                sin_1 = torch.nn.functional.pad(sin_1, (0, pad_size), value=0.0)

        # Apply rotations with matching dimensions
        out1 = x1 * cos_0 - x2 * sin_0
        out2_temp = x1 * sin_1 + x2 * cos_1

        # Truncate back to original size for odd dimensions
        out2 = out2_temp[..., :d2]

        return torch.cat([out1, out2], dim=-1)


class YaRN(RoPE):
    """
    An implementation of YaRN (Yet another RoPE extensioN).
    https://arxiv.org/abs/2309.00071
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig", *args, **kwargs):
        super().__init__(config)

        # YaRN parameters
        self.original_max_position = config.context_length

        # YaRN attention scaling factor (from paper equation)
        factor = (
            4.0  # e.g. if original_max_position is 4096, then a factor of 4.0 is 16384
        )
        self.scale = 0.1 * math.log(factor) + 1.0

        # Generate position-dependent scaling factors for NTK-by-parts
        self.alpha = 1  # threshold for full scaling
        self.beta = 32  # threshold for no scaling

        # Cache buffers
        self.dim_ranges = None
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = None

    def _compute_rope_embeddings(
        self, head_dim, seq_len, device, dtype, offset: int = 0
    ):
        """Compute YaRN-scaled RoPE embeddings with offset and odd dimension support."""
        if (
            self._cached_seq_length is None
            or seq_len > self._cached_seq_length
            or self._cached_cos is None
            or self._cached_cos.device != device
            or self._cached_cos.dtype != dtype
        ):
            # Calculate dimensions for even/odd handling
            freq_dim = (head_dim + 1) // 2  # Ceiling division for odd dimensions
            scales_dim = head_dim // 2  # Floor division for scaling

            if self.dim_ranges is None:
                theta = 10000
                # Create frequencies with ceiling division
                self.inv_freq = 1.0 / (
                    theta ** (torch.arange(0, freq_dim).float() / head_dim)
                ).to(device)

                # Calculate rotations per dimension
                positions = torch.arange(self.original_max_position)
                dim_rotations = positions.unsqueeze(1) * self.inv_freq.unsqueeze(0)
                self.dim_ranges = (dim_rotations[-1] / math.pi) * 2

            # Consider total sequence length including offset for scaling
            total_seq_len = seq_len + offset
            dynamic_scale = max(1.0, total_seq_len / self.original_max_position)

            # Generate position indices with offset
            positions = torch.arange(seq_len, device=device, dtype=dtype) + offset

            # Calculate per-dimension scaling factors (using floor division)
            dim_scales = torch.ones(scales_dim, device=device, dtype=dtype)

            for i in range(scales_dim):
                rotations = self.dim_ranges[i]
                if rotations <= self.alpha:
                    dim_scales[i] = dynamic_scale
                elif rotations >= self.beta:
                    dim_scales[i] = 1.0
                else:
                    dim_scales[i] = 1.0 + (dynamic_scale - 1.0) * (
                        self.beta - rotations
                    ) / (self.beta - self.alpha)

            # Pad dim_scales to match inv_freq if needed
            if freq_dim > scales_dim:
                dim_scales = torch.nn.functional.pad(
                    dim_scales, (0, freq_dim - scales_dim), mode="constant", value=1.0
                )

            # Apply scaling
            scaled_inv_freq = self.inv_freq * dim_scales
            pos_emb = positions.unsqueeze(1) * scaled_inv_freq.unsqueeze(0)
            pos_emb = pos_emb * self.scale

            # Handle odd dimensions in the final reshape
            cos = torch.cos(pos_emb)
            sin = torch.sin(pos_emb)

            # Repeat and reshape, handling odd dimensions
            cos = cos.repeat(1, 2).view(1, 1, seq_len, -1)
            sin = sin.repeat(1, 2).view(1, 1, seq_len, -1)

            # Trim if needed for odd dimensions
            if head_dim % 2 == 1:
                cos = cos[:, :, :, :head_dim]
                sin = sin[:, :, :, :head_dim]

            self._cached_cos = cos.to(dtype)
            self._cached_sin = sin.to(dtype)
            self._cached_seq_length = seq_len


ENCODING_REGISTRY = {"alibi": ALiBi, "nope": NoPE, "rope": RoPE, "yarn": YaRN}
