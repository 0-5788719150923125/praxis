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
        # Number of rotated head-dim slots and which bands are kept, frozen on
        # the first call from the base theta. Per-depth theta deltas then only
        # shift the magnitudes of these bands, never the rotated/unrotated
        # split (see roadmap caveat).
        self._pos_dim: Optional[int] = None
        self._kept_idx: Optional[torch.Tensor] = None

    def _compute_inv_freq(
        self, head_dim: int, device: torch.device, current_depth: int
    ) -> torch.Tensor:
        if self._kept_idx is None:
            # Freeze the rotated/unrotated split from the base theta (depth 0).
            base = super()._compute_inv_freq(head_dim, device, current_depth=0)
            mask = base > self.target_freq
            idx = mask.nonzero(as_tuple=True)[0]
            # If L is so large the threshold drops everything, fall back to
            # the highest-frequency band so we still encode some position.
            if idx.numel() == 0:
                idx = torch.tensor([0], device=device)
            self._kept_idx = idx
            self._pos_dim = 2 * idx.numel()

        full = super()._compute_inv_freq(head_dim, device, current_depth)
        return full[self._kept_idx.to(full.device)]

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
