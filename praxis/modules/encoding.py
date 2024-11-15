import torch
from torch import nn
import torch.nn.functional as F
import math

from transformers import AutoConfig


class NoPE(nn.Module):
    """
    Implementation of NoPE (No Position Encoding) with head-wise attention scaling.
    https://arxiv.org/abs/2404.12224
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.num_dims // config.num_heads
        # Initialize scaling factors - one per head with linspace
        self.head_scales = nn.Parameter(torch.linspace(1.2, 1.2, self.num_heads))

    def before_scores(self, q, k, v):
        # Get base scaling factor
        base_scale = 1.0 / math.sqrt(self.head_dim)

        # Reshape scales for broadcasting
        scaling = self.head_scales.view(1, -1, 1, 1) * base_scale

        # Apply scaling to queries
        return q * scaling, k, v

    def after_scores(self, scores, token_indices):
        return scores


class ALiBi(NoPE):
    """
    This class implements Attention with Linear Biases (ALiBi), which is a form of
    length extrapolation that does not require trainable parameters.
    https://arxiv.org/abs/2108.12409
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        # Pre-compute the ALiBi slopes
        slopes = 2 ** (-8 * torch.arange(1, config.num_heads + 1) / config.num_heads)
        self.register_buffer("slopes", slopes)
        self.register_buffer(
            "positions", torch.arange(config.context_length, dtype=torch.float32)
        )

    def compute_before(self, q, k, v):
        return q, k, v

    def compute_after(self, scores, token_indices):
        batch_size, num_heads, seq_len, _ = scores[0].shape
        if torch.is_tensor(token_indices):
            # If token indices were provided (by a router, perhaps), use them
            positions = self.positions[token_indices]
        else:
            # Else, slice from the pre-computed ALiBi biases
            positions = (
                self.positions[:seq_len].unsqueeze(0).expand(batch_size, seq_len)
            )
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        biases = self.slopes.view(1, num_heads, 1, 1) * pos_diff.unsqueeze(1)
        scores = [score - biases for score in scores]
        return scores


class RoPE(NoPE):
    """
    Implementation of Rotary Position Embeddings (RoPE).
    https://arxiv.org/abs/2104.09864
    """

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        # For differential mode, the actual head dimension is doubled
        self.effective_head_dim = self.head_dim * (2 if config.differential else 1)

        # Important: RoPE operates on pairs of dimensions
        assert self.head_dim % 2 == 0, "Head dimension must be even for RoPE"

        # Generate inverse frequencies - note we use head_dim not effective_head_dim
        # since we'll apply RoPE separately to each half in differential mode
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Cache buffers
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = None

        # Store differential flag
        self.differential = config.differential

    def _compute_rope_embeddings(self, seq_len, device, dtype):
        """Compute sin and cos embeddings."""
        # Recompute if cache is invalid
        if (
            self._cached_seq_length is None
            or seq_len > self._cached_seq_length
            or self._cached_cos is None
            or self._cached_cos.device != device
            or self._cached_cos.dtype != dtype
        ):

            positions = torch.arange(seq_len, device=device)
            # [seq_len, dim/2]
            pos_emb = positions.unsqueeze(1) * self.inv_freq.unsqueeze(0)

            # [1, 1, seq_len, dim]
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

    def before_scores(self, q, k, v):
        # Get sequence length and device
        seq_len = q.size(2)
        device = q.device
        dtype = q.dtype

        # Ensure embeddings are computed and cached
        self._compute_rope_embeddings(seq_len, device, dtype)

        # Get appropriate slice of cached values
        cos = self._cached_cos[:, :, :seq_len, :]
        sin = self._cached_sin[:, :, :seq_len, :]

        if self.differential:
            # Split q and k into their differential halves
            q1, q2 = q.chunk(2, dim=-1)
            k1, k2 = k.chunk(2, dim=-1)

            # Apply RoPE to each half separately
            q1_rope = self._apply_rotary_pos_emb(q1, cos, sin)
            q2_rope = self._apply_rotary_pos_emb(q2, cos, sin)
            k1_rope = self._apply_rotary_pos_emb(k1, cos, sin)
            k2_rope = self._apply_rotary_pos_emb(k2, cos, sin)

            # Recombine the halves
            q_rope = torch.cat([q1_rope, q2_rope], dim=-1)
            k_rope = torch.cat([k1_rope, k2_rope], dim=-1)
        else:
            # Standard RoPE application
            q_rope = self._apply_rotary_pos_emb(q, cos, sin)
            k_rope = self._apply_rotary_pos_emb(k, cos, sin)

        return q_rope, k_rope, v

    def after_scores(self, scores, token_indices):
        return scores
