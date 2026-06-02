"""Energy generative head for CALM.

MLP-based generative head following the reference implementation
(github.com/shaochenze/calm). Maps a conditioning hidden state ``h``
plus uniform noise ``ε`` to a latent proposal ``z_hat``. Trained by
the energy-score loss (section 3.3 of arXiv 2510.27688).

Faithful to the reference on a few points that were biasing training in
earlier versions of this file:
- Conditioning ``y`` is re-injected at *every* ``MLPBlock`` (via concat
  with the LayerNorm'd residual stream), not just at the input.
- Both noise and conditioning paths run through a Linear → LayerNorm
  before any block sees them.
- The final projection is a small (LayerNorm → Linear → SiLU → Linear)
  head whose last Linear is zero-initialised, so the head starts
  predicting near-zero latents and learns to expand from there.
- Noise is uniform [-0.5, 0.5].
"""

import torch
import torch.nn.functional as F
from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.linears = nn.Sequential(
            nn.Linear(2 * channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, 2 * channels, bias=True),
        )
        self.down_proj = nn.Linear(channels, channels, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.linears(torch.cat((self.in_ln(x), y), dim=-1))
        gate, up = h.chunk(2, dim=-1)
        return x + self.down_proj(F.silu(gate) * up)


class FinalLayer(nn.Module):
    def __init__(self, channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.linears = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, out_channels, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linears(self.in_ln(x))


class EnergyHead(nn.Module):
    """Conditioned implicit generator of continuous latents.

    ``hidden_dim`` is the residual width carrying the noise/cond stream
    through the blocks; ``cond_dim`` / ``noise_dim`` are the input
    widths (typically the LM hidden state width and the noise vector
    width respectively).
    """

    def __init__(
        self,
        cond_dim: int,
        noise_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.cond_embd = nn.Linear(cond_dim, hidden_dim, bias=True)
        self.noise_embd = nn.Linear(noise_dim, hidden_dim, bias=True)
        self.norm_cond = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.norm_noise = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.blocks = nn.ModuleList([MLPBlock(hidden_dim) for _ in range(num_blocks)])
        self.final_layer = FinalLayer(hidden_dim, latent_dim)

        # Zero-init the last projection: the head starts predicting
        # near-zero latents and grows from there. Without this the head
        # emits arbitrarily-scaled outputs from random init and has to
        # claw its way back through the energy score.
        nn.init.zeros_(self.final_layer.linears[-1].weight)
        nn.init.zeros_(self.final_layer.linears[-1].bias)

    def forward(self, h: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Generate a latent proposal.

        Args:
            h: ``[..., cond_dim]`` conditioning.
            noise: ``[..., noise_dim]`` noise drawn by the caller.

        Returns:
            ``[..., latent_dim]`` latent sample.
        """
        cond = self.norm_cond(self.cond_embd(h))
        x = self.norm_noise(self.noise_embd(noise))
        for block in self.blocks:
            x = block(x, cond)
        return self.final_layer(x)

    def sample(
        self,
        h: torch.Tensor,
        num_samples: int,
        noise_dtype: torch.dtype = torch.float32,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Draw ``num_samples`` latents per conditioning row.

        Noise is uniform on [-0.5, 0.5]; the energy score is a proper
        scoring rule, so the head learns to map this noise to whatever
        target distribution the loss is fed. ``noise_scale`` shrinks the
        noise toward 0 (the conditional-mean prediction) - generation uses
        it as a temperature, so low T concentrates on the head's best guess.

        Returns ``[num_samples, ..., latent_dim]``.
        """
        shape = h.shape[:-1] + (self.noise_dim,)
        expanded = h.unsqueeze(0).expand(num_samples, *h.shape)
        noise = (
            torch.rand(num_samples, *shape, device=h.device, dtype=noise_dtype) - 0.5
        ) * noise_scale
        return self(expanded, noise)
