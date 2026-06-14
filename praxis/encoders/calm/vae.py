"""Token-chunk VAE for CALM.

Compresses K contiguous tokens into a single continuous latent of size
``latent_dim``, and decodes latents back to K per-token feature vectors
(an external LM head turns those into logits). This is the autoencoder
described in section 3.2 of the CALM paper (arXiv 2510.27688).

The VAE is token-agnostic: it just sees token ids and vocab size, so
CALM can sit on top of any tokenizer (BPE, char, byte).
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class HarmonicDropout(nn.Module):
    """Dropout whose keep-rate is a standing-wave field, not a scalar.

    The drop rate over a ``[..., N, C]`` activation is
    ``base * (1 + (sin(k·n) + sin(k·c))/2)`` - superposed sinusoids across the
    sequence/patch axis N and the channel axis C, so each axis modulates on its
    own (non-flat marginals). ``n_cycles`` full periods span each axis, so the
    frequency is attuned to the input's own extent rather than a fixed step
    count. The field averages to ``base``, so mean regularization is preserved;
    per-element inverted scaling keeps E[output] == input. Inactive in eval
    (so it vanishes once the codec freezes) and identical to ``nn.Dropout``
    when ``base == 0``.
    """

    def __init__(self, base: float, n_cycles: int = 2) -> None:
        super().__init__()
        self.base = float(base)
        self.n_cycles = int(n_cycles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.base <= 0.0 or x.dim() < 2:
            return x
        N, C = x.shape[-2], x.shape[-1]
        tau = 2.0 * math.pi * self.n_cycles
        n = torch.linspace(0.0, 1.0, N, device=x.device, dtype=x.dtype)
        c = torch.linspace(0.0, 1.0, C, device=x.device, dtype=x.dtype)
        field = 0.5 * (
            torch.sin(tau * n)[:, None] + torch.sin(tau * c)[None, :]
        )  # [N,C]
        keep = (1.0 - self.base * (1.0 + field)).clamp(1e-3, 1.0)
        mask = torch.bernoulli(keep.expand_as(x))
        return x * mask / keep


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block (the reference's AELayer shape).

    ``x + drop(W2(SiLU(W1(RMSNorm(x)))))``. Residual + pre-norm is what lets
    the codec stack deepen without the vanishing-gradient stall a plain
    Linear/SiLU stack hits, so capacity scales with ``depth``.
    """

    def __init__(self, dim: int, drop: nn.Module) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.drop = drop
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.fc2(self.act(self.fc1(self.norm(x)))))


class CALMVAE(nn.Module):
    """Chunked token VAE.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Per-token embedding dim inside the VAE.
        chunk_size: K tokens per latent.
        latent_dim: Continuous latent dim.
        hidden_dim: Width of the encoder / decoder MLPs.
        depth: Number of residual blocks per side. Higher = more codec
            capacity (the reference reaches recon CE ~0.04 with a residual
            stack; a flat 2-layer MLP stalls ~0.3).
        latent_norm: Fix the latent to unit per-dim RMS (norm = sqrt(D))
            before it is decoded. Pins the latent geometry so it can't drift
            into the large-norm / tiny-variance brittleness that makes the
            energy head's target unreachably precise. Parameter-free, so the
            geometry is stationary across the stage-1 -> stage-2 freeze.
        dropout: Dropout rate, applied at three sites as in the reference:
            input token ids (zeroed), the sampled latent z, and inside the
            encoder / decoder blocks. The first two are load-bearing for
            generation: they train the decoder to map a NEIGHBORHOOD of z
            to the right tokens, so the LM head's imperfect latent
            predictions still decode to text.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        chunk_size: int,
        latent_dim: int,
        hidden_dim: int,
        depth: int = 2,
        latent_norm: bool = False,
        dropout: float = 0.15,
        dropout_mode: str = "scalar",
        dropout_cycles: int = 2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.latent_norm = bool(latent_norm)
        self.dropout_p = float(dropout)

        def _drop():
            if dropout_mode == "harmonic":
                return HarmonicDropout(dropout, n_cycles=dropout_cycles)
            return nn.Dropout(dropout)

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)

        # Encoder: project K token embeddings to hidden, refine through a
        # residual block stack, then project to posterior params.
        self.enc_in = nn.Linear(chunk_size * embed_dim, hidden_dim)
        self.enc_blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, _drop()) for _ in range(depth)]
        )
        # Norm before posterior projection: keeps μ/logvar in a well-scaled
        # range and matches the reference's LlamaRMSNorm in the AE encoder.
        self.params_norm = nn.RMSNorm(hidden_dim)
        self.to_params = nn.Linear(hidden_dim, 2 * latent_dim)

        # Decoder: latent to hidden, residual block stack, expand to K
        # per-token feature vectors.
        self.dec_in = nn.Linear(latent_dim, hidden_dim)
        self.dec_blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, _drop()) for _ in range(depth)]
        )
        self.dec_expand = nn.Linear(hidden_dim, chunk_size * hidden_dim)
        # Norm before the classifier consumes decoder features.
        self.out_norm = nn.RMSNorm(hidden_dim)

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

        if self.training and self.dropout_p > 0:
            # Reference-faithful input corruption: random ids -> 0, forcing
            # the latent to denoise rather than memorize exact patches.
            keep = torch.rand_like(input_ids, dtype=torch.float) > self.dropout_p
            input_ids = input_ids * keep.long()

        emb = self.tok_emb(input_ids)  # [B, N*K, E]
        emb = emb.view(B, N, K * self.embed_dim)
        h = self.enc_in(emb)  # [B, N, H]
        for blk in self.enc_blocks:
            h = blk(h)
        h = self.params_norm(h)
        params = self.to_params(h)  # [B, N, 2L]
        mean, logvar = params.chunk(2, dim=-1)
        # Bound logvar to keep KL finite and prevent posterior collapse
        # via degenerate variances.
        logvar = logvar.clamp(min=-10.0, max=10.0)
        return mean, logvar

    def reparameterize(
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mean + std * torch.randn_like(std)

    def normalize_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Fix the latent to unit per-dim RMS (norm = sqrt(D)). Parameter-free
        so the geometry stays stationary once the codec freezes. No-op unless
        ``latent_norm`` is set."""
        if not self.latent_norm:
            return x
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)

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
        # Single source of truth for "the decoder consumes a normalized
        # latent": covers teacher-forced recon, the energy head's zero-noise
        # decode, and generation, all of which route through here.
        z = self.normalize_latent(z)
        # Latent dropout (reference-faithful): the decoder learns to decode
        # perturbed latents, the robustness generation depends on. Inactive
        # in eval, so the frozen stage-2 codec and generation see clean z.
        z = F.dropout(z, p=self.dropout_p, training=self.training)
        h = self.dec_in(z)  # [B, N, H]
        for blk in self.dec_blocks:
            h = blk(h)
        h = self.dec_expand(h)  # [B, N, K*H]
        h = h.view(B, N, K, self.hidden_dim)
        h = h.reshape(B, N * K, self.hidden_dim)
        return self.out_norm(h)

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
