"""Token-chunk VAE for CALM.

Compresses K contiguous tokens into a single continuous latent of size
``latent_dim``, and decodes latents back to K per-token feature vectors
(an external LM head turns those into logits). This is the autoencoder
described in section 3.2 of the CALM paper (arXiv 2510.27688).

The VAE is token-agnostic: it just sees token ids and vocab size, so
CALM can sit on top of any tokenizer (BPE, char, byte).
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class CALMVAE(nn.Module):
    """Chunked token VAE.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Per-token embedding dim inside the VAE.
        chunk_size: K tokens per latent.
        latent_dim: Continuous latent dim.
        hidden_dim: Width of the encoder / decoder MLPs.
        dropout: Dropout inside the encoder / decoder MLPs.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        chunk_size: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)

        self.encoder_mlp = nn.Sequential(
            nn.Linear(chunk_size * embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.to_params = nn.Linear(hidden_dim, 2 * latent_dim)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, chunk_size * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def encode(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode token ids into posterior parameters.

        Args:
            input_ids: ``[B, N*K]`` token ids (already padded to a
                multiple of ``chunk_size``).

        Returns:
            ``(mean, logvar)`` each of shape ``[B, N, latent_dim]``.
        """
        B, L = input_ids.shape
        K = self.chunk_size
        assert L % K == 0, f"seq len {L} not divisible by chunk size {K}"
        N = L // K

        emb = self.tok_emb(input_ids)  # [B, N*K, E]
        emb = emb.view(B, N, K * self.embed_dim)
        h = self.encoder_mlp(emb)  # [B, N, H]
        params = self.to_params(h)  # [B, N, 2L]
        mean, logvar = params.chunk(2, dim=-1)
        # Bound logvar to keep KL finite and prevent posterior collapse
        # via degenerate variances.
        logvar = logvar.clamp(min=-10.0, max=10.0)
        return mean, logvar

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mean + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to per-token decoder features.

        Args:
            z: ``[B, N, latent_dim]`` latent samples.

        Returns:
            ``[B, N*K, hidden_dim]`` decoder hidden states in patch-major
            order (all K tokens of patch 0, then patch 1, ...). The token
            classifier that turns these into logits is owned externally
            (the injected LM head), so CALM can swap forward/crystal/etc.
        """
        B, N, _ = z.shape
        K = self.chunk_size
        h = self.decoder_mlp(z)  # [B, N, K*H]
        h = h.view(B, N, K, self.hidden_dim)
        return h.reshape(B, N * K, self.hidden_dim)

    @staticmethod
    def kl_divergence(
        mean: torch.Tensor, logvar: torch.Tensor, per_dim_clip: float = 0.0
    ) -> torch.Tensor:
        """Per-position KL against N(0, I), optionally clipped per dim.

        ``per_dim_clip`` implements the paper's "free bits" regulariser
        (section 3.2): individual latent dims contribute at least this
        many nats of KL before the loss rewards reducing them further.
        """
        per_dim = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar)
        if per_dim_clip and per_dim_clip > 0.0:
            per_dim = per_dim.clamp(min=per_dim_clip)
        return per_dim.sum(dim=-1)  # [B, N]
