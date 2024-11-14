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

    def compute_before(self, q, k, v, token_indices):
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

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        # Cache sin/cos tables
        self.register_buffer(
            "positions", torch.arange(config.context_length, dtype=torch.float32)
        )
        # Generate frequency bands
        theta = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("theta", theta)

    def before_scores(self, q, k, v, token_indices):
        # Get sequence length
        seq_len = q.size(2)

        # Compute rotary embeddings for the sequence length
        if torch.is_tensor(token_indices):
            positions = self.positions[token_indices]
        else:
            positions = self.positions[:seq_len]

        cos, sin = self._compute_rope_embeddings(positions)

        # Apply rotary embeddings to queries and keys
        q_rope, k_rope = self._apply_rotary_pos_emb(q, k, cos, sin)

        return q_rope, k_rope, v

    def after_scores(self, scores, token_indices):
        return scores

    def _compute_rope_embeddings(self, positions):
        # positions shape: (seq_len,)
        # theta shape: (head_dim/2,)
        freqs = torch.outer(positions, self.theta)  # (seq_len, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)

        # Complex rotation
        cos = torch.cos(emb)  # (seq_len, head_dim)
        sin = torch.sin(emb)  # (seq_len, head_dim)
        return cos, sin

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        # Reshape cos/sin for broadcasting:
        # (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
        cos = cos.view(1, 1, cos.shape[0], cos.shape[1])
        sin = sin.view(1, 1, sin.shape[0], sin.shape[1])

        # Apply rotary embeddings
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed
