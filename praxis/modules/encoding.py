import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig


class NoPE(nn.Module):
    """
    Implementation of NoPE (No Position Encoding) with head-wise attention scaling.
    https://arxiv.org/abs/2404.12224
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig, scaled: bool = False):
        super().__init__()
        self.num_query_heads = config.num_heads * config.num_queries
        self.multiplier = 2 if config.differential else 1
        self.head_dim = config.num_dims // config.num_heads // self.multiplier
        # Initialize scaling factors - one per head with linspace
        self.scaled = scaled
        if self.scaled:
            self.head_scales = nn.Parameter(
                torch.linspace(1.2, 1.2, self.num_query_heads)
            )

    def before_scores(self, q, k, v, offset: int = 0):
        if self.scaled:
            # Get base scaling factor
            base_scale = 1.0 / math.sqrt(self.head_dim)

            # Reshape scales for broadcasting
            scaling = self.head_scales.view(1, -1, 1, 1) * base_scale

            # Apply scaling to queries
            return q * scaling, k, v
        else:
            return q, k, v

    def after_scores(self, scores, offset: int = 0):
        return scores


class ALiBi(NoPE):
    """
    This class implements Attention with Linear Biases (ALiBi), which is a form of
    length extrapolation that does not require trainable parameters.
    https://arxiv.org/abs/2108.12409
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig, *args, **kwargs):
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

    def __init__(self, config: AutoConfig, *args, **kwargs):
        super().__init__(config)
        # Important: RoPE operates on pairs of dimensions
        assert self.head_dim % 2 == 0, "Head dimension must be even for RoPE"

        # Generate inverse frequencies for base head_dim
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Linear scaling
        self.scale = 1.0

        # Cache buffers
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = None

    def before_scores(self, q, k, v, offset: int = 0):
        seq_len = q.size(2)
        device = q.device
        dtype = q.dtype

        # Pass offset to compute_rope_embeddings
        self._compute_rope_embeddings(seq_len, device, dtype, offset)

        cos = self._cached_cos[:, :, :seq_len, :]
        sin = self._cached_sin[:, :, :seq_len, :]

        q_chunks = q.chunk(q.size(-1) // self.head_dim, dim=-1)
        k_chunks = k.chunk(k.size(-1) // self.head_dim, dim=-1)

        q_rope_chunks = [
            self._apply_rotary_pos_emb(q_chunk, cos, sin) for q_chunk in q_chunks
        ]
        k_rope_chunks = [
            self._apply_rotary_pos_emb(k_chunk, cos, sin) for k_chunk in k_chunks
        ]

        q_rope = (
            torch.cat(q_rope_chunks, dim=-1) if len(q_chunks) > 1 else q_rope_chunks[0]
        )
        k_rope = (
            torch.cat(k_rope_chunks, dim=-1) if len(k_chunks) > 1 else k_rope_chunks[0]
        )

        return q_rope, k_rope, v

    def after_scores(self, scores, offset: int = 0):
        return scores

    def _compute_rope_embeddings(self, seq_len, device, dtype, offset: int = 0):
        """Compute sin and cos embeddings with offset."""
        if (
            self._cached_seq_length is None
            or seq_len > self._cached_seq_length
            or self._cached_cos is None
            or self._cached_cos.device != device
            or self._cached_cos.dtype != dtype
        ):
            # Add offset to positions
            positions = (torch.arange(seq_len, device=device) + offset) * self.scale
            pos_emb = positions.unsqueeze(1) * self.inv_freq.unsqueeze(0)

            cos = torch.cos(pos_emb).repeat(1, 1, 1, 2).view(1, 1, seq_len, -1)
            sin = torch.sin(pos_emb).repeat(1, 1, 1, 2).view(1, 1, seq_len, -1)

            self._cached_cos = cos.to(dtype)
            self._cached_sin = sin.to(dtype)
            self._cached_seq_length = seq_len

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self, x, cos, sin):
        return (x * cos) + (self._rotate_half(x) * sin)


class YaRN(RoPE):
    """
    An implementation of YaRN (Yet another RoPE extensioN).
    https://arxiv.org/abs/2309.00071
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig, *args, **kwargs):
        super().__init__(config)
        assert self.head_dim % 2 == 0, "Head dimension must be even for RoPE"

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

        # Calculate rotations per dimension
        positions = torch.arange(self.original_max_position)
        dim_rotations = positions.unsqueeze(1) * self.inv_freq.unsqueeze(
            0
        )  # [seq_len, dim/2]
        self.dim_ranges = (dim_rotations[-1] / math.pi) * 2  # rotations at max position

        # Cache buffers
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = None

    def _compute_rope_embeddings(self, seq_len, device, dtype, offset: int = 0):
        """Compute YaRN-scaled RoPE embeddings with offset."""
        if (
            self._cached_seq_length is None
            or seq_len > self._cached_seq_length
            or self._cached_cos is None
            or self._cached_cos.device != device
            or self._cached_cos.dtype != dtype
        ):
            # Consider total sequence length including offset for scaling
            total_seq_len = seq_len + offset
            dynamic_scale = max(1.0, total_seq_len / self.original_max_position)

            # Generate position indices with offset
            positions = torch.arange(seq_len, device=device, dtype=dtype) + offset

            # Calculate per-dimension scaling factors
            dim_scales = torch.ones(self.head_dim // 2, device=device, dtype=dtype)

            for i in range(self.head_dim // 2):
                rotations = self.dim_ranges[i]
                if rotations <= self.alpha:
                    dim_scales[i] = dynamic_scale
                elif rotations >= self.beta:
                    dim_scales[i] = 1.0
                else:
                    dim_scales[i] = 1.0 + (dynamic_scale - 1.0) * (
                        self.beta - rotations
                    ) / (self.beta - self.alpha)

            scaled_inv_freq = self.inv_freq * dim_scales
            pos_emb = positions.unsqueeze(1) * scaled_inv_freq.unsqueeze(0)
            pos_emb = pos_emb * self.scale

            cos = torch.cos(pos_emb).repeat(1, 1, 1, 2).view(1, 1, seq_len, -1)
            sin = torch.sin(pos_emb).repeat(1, 1, 1, 2).view(1, 1, seq_len, -1)

            self._cached_cos = cos.to(dtype)
            self._cached_sin = sin.to(dtype)
            self._cached_seq_length = seq_len


ENCODING_REGISTRY = {"alibi": ALiBi, "nope": NoPE, "rope": RoPE, "yarn": YaRN}
