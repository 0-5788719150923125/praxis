import math
from typing import Optional, TypeVar

import torch

from praxis.encoding.rope import RoPE

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class HoPE(RoPE):
    """
    High-frequency rotary Position Embedding.
    https://arxiv.org/abs/2410.21216

    Drops RoPE bands slower than one full cycle over the training context
    length L (inv_freq < 2*pi/L) and leaves the matching head-dim slots
    unrotated. Those slots become a position-free content channel; the
    remaining high-frequency slots still carry relative position via the
    usual rotation. Authors report large length-extrapolation and in-context
    copying gains over RoPE at the same parameter count.
    """

    def __init__(self, config: ConfigType, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # The threshold is defined against the training context length, not
        # max positional capacity. block_size is what the data module
        # actually produces sequences at.
        self.context_length: int = config.block_size
        self.target_freq: float = 2.0 * math.pi / self.context_length
        # Number of rotated head-dim slots, resolved on first call once we
        # know head_dim.
        self._pos_dim: Optional[int] = None

    def _compute_rope_embeddings(
        self,
        head_dim: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        offset: int = 0,
        block_ids: Optional[torch.Tensor] = None,
    ) -> None:
        if self.inv_freq is None:
            full_inv_freq = 1.0 / (
                self.theta
                ** (
                    2 * torch.arange(0, head_dim // 2, device=device).float() / head_dim
                )
            )
            mask = full_inv_freq > self.target_freq
            kept = full_inv_freq[mask]
            # If L is so large the threshold drops everything, fall back to
            # the highest-frequency band so we still encode some position.
            if kept.numel() == 0:
                kept = full_inv_freq[:1]
            self.inv_freq = kept
            self._pos_dim = 2 * self.inv_freq.numel()

        if block_ids is not None and block_ids.size(1) != 1:
            positions = self._compute_relative_positions_vectorized(block_ids, device)
        else:
            positions = (torch.arange(seq_len, device=device) + offset).unsqueeze(0)

        freqs = torch.einsum("bi,j->bij", positions, self.inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        cos = torch.stack([cos, cos], dim=-1).view(cos.size(0), 1, cos.size(1), -1)
        sin = torch.stack([-sin, sin], dim=-1).view(sin.size(0), 1, sin.size(1), -1)

        self._cached_cos = cos.to(dtype)
        self._cached_sin = sin.to(dtype)
        self._cached_seq_length = seq_len

    def _apply_rotary_pos_emb(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Rotate the first pos_dim head-dim slots; pass the rest through."""
        pos_dim = self._pos_dim
        head_dim = x.size(-1)
        if pos_dim is None or pos_dim >= head_dim:
            # No truncation needed; vanilla rotation.
            return super()._apply_rotary_pos_emb(x, cos, sin)

        x_rot = super()._apply_rotary_pos_emb(x[..., :pos_dim], cos, sin)
        return torch.cat([x_rot, x[..., pos_dim:]], dim=-1)
