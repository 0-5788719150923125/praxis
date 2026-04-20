"""Energy generative head for CALM.

Maps a conditioning hidden state ``h`` plus a random noise vector to a
latent proposal ``z_hat``. Trained by the energy-score loss (section 3.3
of the CALM paper) so that the implicit distribution over ``z_hat`` at
fixed ``h`` matches the posterior over the next chunk latent.
"""

import torch
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x)))


class EnergyBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class EnergyHead(nn.Module):
    """Conditioned implicit generator of continuous latents.

    Architecture: project (h_proj + noise_proj) to a residual stream of
    width ``hidden_dim`` via L SwiGLU blocks, then project to
    ``latent_dim``.
    """

    def __init__(
        self,
        cond_dim: int,
        noise_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_blocks: int = 3,
        mlp_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.cond_proj = nn.Linear(cond_dim, hidden_dim, bias=False)
        self.noise_proj = nn.Linear(noise_dim, hidden_dim, bias=False)
        self.blocks = nn.ModuleList(
            [EnergyBlock(hidden_dim, hidden_dim * mlp_mult, dropout) for _ in range(num_blocks)]
        )
        self.norm = nn.RMSNorm(hidden_dim)
        self.to_z = nn.Linear(hidden_dim, latent_dim, bias=False)

    def forward(self, h: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Generate a latent proposal.

        Args:
            h: ``[..., cond_dim]`` conditioning (e.g. LM hidden state at
                the position that should predict the next latent).
            noise: ``[..., noise_dim]`` random noise drawn fresh each
                call.

        Returns:
            ``[..., latent_dim]`` latent sample.
        """
        x = self.cond_proj(h) + self.noise_proj(noise)
        for block in self.blocks:
            x = block(x)
        return self.to_z(self.norm(x))

    def sample(
        self,
        h: torch.Tensor,
        num_samples: int,
        noise_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Draw ``num_samples`` latents for each conditioning row.

        Args:
            h: ``[..., cond_dim]`` conditioning.
            num_samples: How many independent samples per position.

        Returns:
            ``[num_samples, ..., latent_dim]`` latents.
        """
        shape = h.shape[:-1] + (self.noise_dim,)
        expanded = h.unsqueeze(0).expand(num_samples, *h.shape)
        noise = torch.randn(
            num_samples, *shape, device=h.device, dtype=noise_dtype
        )
        return self(expanded, noise)
