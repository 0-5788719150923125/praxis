import math
from typing import Optional, Tuple, TypeVar

import torch
from torch import nn

from praxis.encoding.nope import NoPE
from praxis.encoding.rope import THETA_INIT, segment_positions

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class LearnedRateRoPE(NoPE):
    """Rotary position with a learned rate for every rotary pair of every KV head.

    Rates start on the geometric rotary schedule (base ``THETA_INIT``) and are
    held in log space, so they stay positive. A zero-init per-depth shift moves
    every rate of one recurrent pass together. Kuramoto Attention
    (arXiv:2606.11585) reads these rates as each oscillator's natural frequency:
    the phase a token gains per position before two tokens are compared.

    Pairs use the rotate-half layout (``x[..., j]`` with ``x[..., j + d/2]``) and
    pair ``j`` turns at rate ``j``, so every learned rate reaches the scores.
    Query heads beyond the KV head count (GQA) use their KV head's rates.
    """

    metric_descriptions = {
        "rope_rate_mean": {
            "description": (
                "Mean learned rotary rate. Faster rates decorrelate scores over "
                "fewer positions; the paper saw the mean fall with depth (later "
                "layers coupling longer-range)."
            ),
            "chart": {
                "title": "Mean Rotary Rate",
                "y_label": "radians per position",
                "y_scale": "logarithmic",
                "group": "kuramoto",
                "group_order": 40,
                "order": 80,
            },
        },
    }
    per_layer_metrics = frozenset(metric_descriptions)

    def __init__(self, config: ConfigType, *args, **kwargs):
        super().__init__(config)
        head_dim = config.head_size or config.hidden_size // config.num_heads
        if head_dim % 2:
            raise ValueError(
                f"rope_learned_rates rotates pairs, so the head width must be "
                f"even (got {head_dim})"
            )
        self.depth = config.depth
        exponent = 2 * torch.arange(head_dim // 2).float() / head_dim
        log_rates = -exponent * math.log(THETA_INIT)
        self.log_rates = nn.Parameter(log_rates.repeat(config.num_heads, 1))
        # Flat [depth], for the reason RoPE's depth_log_theta is flat.
        self.depth_log_rate = nn.Parameter(torch.zeros(self.depth))

    def rates(self, current_depth: int = 0) -> torch.Tensor:
        """``[num_heads, head_dim // 2]`` rates for one recurrent pass."""
        idx = int(current_depth) % self.depth
        return torch.exp(self.log_rates + self.depth_log_rate[idx])

    def before_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        offset: int = 0,
        block_ids: Optional[torch.Tensor] = None,
        current_depth: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rates = self.rates(current_depth)
        heads, bands = rates.shape
        if k.size(1) != heads or q.size(-1) != 2 * bands or q.size(1) % heads:
            raise ValueError(
                f"rope_learned_rates was built for {heads} KV heads of width "
                f"{2 * bands}, got q {tuple(q.shape)} and k {tuple(k.shape)}"
            )
        q_len, k_len = q.size(2), k.size(2)
        if block_ids is not None and block_ids.size(1) != 1:
            positions = segment_positions(block_ids)
        else:
            positions = (
                torch.arange(max(q_len, k_len), device=q.device) + offset
            ).unsqueeze(0)
        # [batch, heads, positions, bands], in float32 for long positions
        angles = positions.float()[:, None, :, None] * rates.float()[None, :, None, :]

        k_angles = angles[:, :, :k_len]
        # Cached decode: the single query sits at the last key position.
        q_angles = angles[:, :, -1:] if q_len == 1 and k_len > 1 else angles[:, :, :q_len]
        group = q.size(1) // heads
        if group > 1:
            q_angles = q_angles.repeat_interleave(group, dim=1)
        return _rotate(q, q_angles), _rotate(k, k_angles), v

    def training_metrics(self) -> dict:
        with torch.no_grad():
            return {"rope_rate_mean": self.rates(0).mean().item()}


def _rotate(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Turn each rotate-half pair of ``x`` by its angle."""
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
