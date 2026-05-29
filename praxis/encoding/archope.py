from typing import TypeVar

import torch
from torch import nn
from torch.distributions.exponential import Exponential

from praxis.activations.serpent import INV_FLOOR_EPS
from praxis.encoding.hope import HoPE

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class ArcHoPE(HoPE):
    """HoPE whose rotary phase is a per-band Serpent warp over (position, depth).

    Mirrors ArcAttention / ArcGLU: each rotated band gets its own learned
    Serpent shape, and recurrent depth couples in through the phase. For band
    i at position p and depth t:

        u_i = p + rho_i * t
        w_i = u_i + lambda_i * (sin^2(alpha_i * u_i) * inv(alpha_i)
                                + gamma_i * sin(beta_i * u_i))
        angle_i = inv_freq[i] * w_i

    The +u_i term keeps the phase monotone in position; the sine terms add
    learned periodicity and harmonics. lambda and rho are zero-init, so at
    step 0 this is exactly learnable-theta HoPE and it specializes from there.

    Trade-off: once the warp activates the phase depends on absolute position,
    so RoPE's strict relative-position property relaxes (as in CARoPE/PaTH).

    Follow-up: per-band specialization diagnostics aren't surfaced yet -
    encoding modules aren't walked by the Arc metric collector.
    """

    def __init__(self, config: ConfigType, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        head_dim = config.head_size or config.hidden_size // config.num_heads
        # Trigger HoPE's band selection now (at THETA_INIT) so the kept-band
        # count is known and the warp params can be sized eagerly.
        self._compute_inv_freq(head_dim, torch.device("cpu"), current_depth=0)
        num_bands = self._pos_dim // 2

        exp = Exponential(torch.tensor(1.0))
        self.warp_alpha = nn.Parameter(exp.sample((num_bands,)))
        self.warp_beta = nn.Parameter(exp.sample((num_bands,)))
        self.warp_gamma = nn.Parameter(torch.empty(num_bands).uniform_(-0.1, 0.1))
        # Zero-init: warp is identity until these grow, so we start as HoPE.
        self.warp_lambda = nn.Parameter(torch.zeros(num_bands))
        self.warp_rho = nn.Parameter(torch.zeros(num_bands))

    def _position_freqs(
        self,
        positions: torch.Tensor,
        inv_freq: torch.Tensor,
        current_depth: int,
    ) -> torch.Tensor:
        # positions [b, p] -> u [b, p, bands]; depth shifts the diagonal phase.
        u = positions.float().unsqueeze(-1) + self.warp_rho * float(current_depth)
        inv_alpha = self.warp_alpha / (self.warp_alpha**2 + INV_FLOOR_EPS**2)
        warp = (
            torch.sin(self.warp_alpha * u).square() * inv_alpha
            + self.warp_gamma * torch.sin(self.warp_beta * u)
        )
        w = u + self.warp_lambda * warp
        return w * inv_freq
