"""CALM encoder: VAE + energy head, exposed via the Praxis encoder interface.

See ``PLAN.md`` for the full architecture rationale. Briefly:

1. ``encode`` compresses K-token chunks to continuous latents. The
   posterior sample ``z`` is projected to ``hidden_size`` and fed to the
   global transformer as the "patch embedding" sequence. The VAE's own
   reconstruction + KL objective is returned as ``encoder_loss`` and
   composes with the rest of the loss container.
2. The global transformer autoregresses over latents.
3. ``decode`` uses the LM hidden state at position ``p`` to drive an
   energy head that produces proposals for latent ``p+1``; those are
   compared against the posterior samples of ``p+1`` under the
   energy-score loss. The reconstructed-token logits are returned for
   sanity checking but do not participate in the main loss (the encoder
   sets ``handles_loss=True`` to bypass the default CE path).
"""

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from praxis.heads.energy import EnergyHead
from praxis.losses.energy_score import energy_score_loss

from .vae import CALMVAE


class CALMEncoder(nn.Module):
    """CALM autoencoder + energy head, plugged into the encoder slot.

    The encoder owns its loss bookkeeping; see ``handles_loss``.
    """

    def __init__(
        self,
        config,
        *,
        chunk_size: int = 8,
        latent_dim: int = 128,
        ae_hidden: int = 512,
        kl_beta: float = 1e-3,
        kl_clip: float = 0.5,
        ae_dropout: float = 0.15,
        noise_dim: int = 128,
        energy_blocks: int = 3,
        energy_samples_n: int = 8,
        energy_samples_m: int = 100,
        energy_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = config

        self.K = chunk_size
        self.latent_dim = latent_dim
        self.ae_hidden = ae_hidden
        self.kl_beta = kl_beta
        self.kl_clip = kl_clip
        self.ae_dropout = ae_dropout

        self.noise_dim = noise_dim
        self.energy_blocks = energy_blocks
        self.energy_samples_n = energy_samples_n
        self.energy_samples_m = energy_samples_m
        self.energy_alpha = energy_alpha

        self.pad_token_id = int(getattr(config, "pad_token_id", 0))

        self.vae = CALMVAE(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_size,
            chunk_size=self.K,
            latent_dim=self.latent_dim,
            hidden_dim=self.ae_hidden,
            dropout=self.ae_dropout,
        )

        # Latent → hidden_size projection for the global transformer.
        self.latent_in = nn.Linear(self.latent_dim, config.hidden_size, bias=False)

        self.energy_head = EnergyHead(
            cond_dim=config.hidden_size,
            noise_dim=self.noise_dim,
            latent_dim=self.latent_dim,
            hidden_dim=max(config.hidden_size, self.latent_dim),
            num_blocks=self.energy_blocks,
            dropout=self.ae_dropout,
        )

        # Loss side-channel: PraxisModel consumes these after decode().
        self._pending_losses: Dict[str, torch.Tensor] = {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"K={self.K}, "
            f"latent={self.latent_dim}, "
            f"ae_hidden={self.ae_hidden}, "
            f"energy_blocks={self.energy_blocks}, "
            f"N={self.energy_samples_n}, "
            f"M={self.energy_samples_m})"
        )

    # ------------------------------------------------------------------
    # Encoder-interface surface
    # ------------------------------------------------------------------

    @property
    def classifier(self) -> nn.Module:
        """LM head used by cut-cross-entropy paths."""
        return self.vae.lm_head

    @property
    def outputs_are_aligned(self) -> bool:
        """The reconstructed logits are returned aligned token-for-token,
        but the main CE path is bypassed via ``handles_loss``."""
        return True

    @property
    def handles_loss(self) -> bool:
        """Skip the default CE path; we register losses internally."""
        return True

    @property
    def sequence_length_multiplier(self) -> int:
        """Global transformer sees patches, not tokens, so the decoder
        side is not a multiplier of the user-supplied ``block_size``."""
        return 1

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def _pad_to_chunk(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        rem = L % self.K
        if rem == 0:
            return input_ids
        pad_len = self.K - rem
        pad = input_ids.new_full((B, pad_len), self.pad_token_id)
        return torch.cat([input_ids, pad], dim=1)

    def encode(
        self, input_ids: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """See ``EncoderInterface``.

        Returns ``(patch_embeds, h_encoder, patch_lengths, block_ids,
        encoder_loss, local_decoder_tokens)``. ``h_encoder`` is ``None``
        because CALM does not have a byte-level local encoder.
        """
        padded = self._pad_to_chunk(input_ids)
        B, L = padded.shape
        N = L // self.K

        mean, logvar = self.vae.encode(padded)  # [B, N, latent_dim]
        z = self.vae.reparameterize(mean, logvar)

        recon_logits, _ = self.vae.decode(z)  # [B, N*K, V]
        recon_loss = F.cross_entropy(
            recon_logits.reshape(-1, self.vae.vocab_size),
            padded.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        kl = self.vae.kl_divergence(mean, logvar, per_dim_clip=self.kl_clip).mean()
        encoder_loss = recon_loss + self.kl_beta * kl

        # Stash what decode() needs.
        self._last_padded = padded
        self._last_mean = mean
        self._last_logvar = logvar
        self._last_z = z
        self._last_N = N

        # Global transformer inputs: one token per patch.
        patch_embeds = self.latent_in(z)  # [B, N, hidden_size]
        patch_lengths = padded.new_full((B, N), self.K, dtype=torch.long)
        # No per-patch sequence boundaries in v1.
        block_ids = padded.new_zeros((B, N), dtype=torch.long) + 1

        return patch_embeds, None, patch_lengths, block_ids, encoder_loss, padded

    def decode(
        self,
        h: torch.Tensor,
        h_encoder: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        patch_lengths: Optional[torch.Tensor] = None,
        local_decoder_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstructed token logits and register the energy loss.

        ``h`` is the global transformer's hidden state per-patch
        (``[B, N, hidden_size]``). The standard reconstruction path uses
        the posterior sample stashed in ``encode``; the energy path
        derives next-latent proposals from ``h[:, :-1]`` and compares
        them against posterior samples of patch ``[:, 1:]``.
        """
        z = self._last_z  # posterior sample used for recon logits
        recon_logits, recon_hidden = self.vae.decode(z)

        if self.training and h.size(1) >= 2:
            self._register_energy_loss(h)

        return recon_logits, recon_hidden

    def _register_energy_loss(self, h: torch.Tensor) -> None:
        """Energy-score loss between next-position model samples and
        next-position posterior samples. See module docstring."""
        B, N = h.shape[0], h.shape[1]
        if N < 2:
            return

        # Position p predicts latent at p+1. h_cond: [B, N-1, hidden]
        h_cond = h[:, :-1, :]

        # Target: posterior samples for patches 1..N-1. M independent
        # draws per position, stop-grad so the VAE does not see the
        # energy path.
        mean_t = self._last_mean[:, 1:, :].detach()
        logvar_t = self._last_logvar[:, 1:, :].detach()
        M = self.energy_samples_m
        std_t = (0.5 * logvar_t).exp()
        eps_t = torch.randn(
            M, *mean_t.shape, device=mean_t.device, dtype=mean_t.dtype
        )
        # [M, B, N-1, L] -> [B, N-1, M, L]
        target_samples = (mean_t.unsqueeze(0) + std_t.unsqueeze(0) * eps_t).permute(
            1, 2, 0, 3
        )

        # Model samples: N draws from energy head. Same reshape.
        N_samples = self.energy_samples_n
        # [N, B, N-1, L] -> [B, N-1, N, L]
        model_raw = self.energy_head.sample(h_cond, num_samples=N_samples).permute(
            1, 2, 0, 3
        )

        loss = energy_score_loss(model_raw, target_samples)
        self._pending_losses["energy"] = self.energy_alpha * loss

    # ------------------------------------------------------------------
    # Loss side-channel
    # ------------------------------------------------------------------

    def consume_pending_losses(self) -> Dict[str, torch.Tensor]:
        """Pop any losses registered during the last ``decode`` call."""
        out = self._pending_losses
        self._pending_losses = {}
        return out
