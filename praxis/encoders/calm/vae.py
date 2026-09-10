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

from praxis.activations import build_activation


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

    ``x + drop(W2(act(W1(RMSNorm(x)))))``. Residual + pre-norm is what lets
    the codec stack deepen without the vanishing-gradient stall a plain
    Linear/act stack hits, so capacity scales with ``depth``. ``activation``
    names the inner nonlinearity (``config.activation``, defaulting to SiLU
    for reference parity).
    """

    def __init__(self, dim: int, drop: nn.Module, activation: str = "silu") -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = build_activation(activation)
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
        activation: str = "silu",
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
            [ResidualMLPBlock(hidden_dim, _drop(), activation) for _ in range(depth)]
        )
        # Norm before posterior projection: keeps μ/logvar in a well-scaled
        # range and matches the reference's LlamaRMSNorm in the AE encoder.
        self.params_norm = nn.RMSNorm(hidden_dim)
        self.to_params = nn.Linear(hidden_dim, 2 * latent_dim)

        # Decoder: latent to hidden, residual block stack, expand to K
        # per-token feature vectors.
        self.dec_in = nn.Linear(latent_dim, hidden_dim)
        self.dec_blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, _drop(), activation) for _ in range(depth)]
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

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
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


class PatchVAE(nn.Module):
    """CALM's autoencoder, over patch FEATURES instead of token chunks.

    ``CALMVAE`` encodes token ids and decodes to K per-token features, because in
    the reference the VAE performs the compression. In the Abstractinator that job
    is already done - the local encoder produces one feature vector per patch before
    the bottleneck - so a second token-chunk VAE would be redundant. What is NOT
    redundant is everything else the VAE provides:

      1. A CONTINUOUS, KL-REGULARIZED, UNIT-SCALE, STATIONARY latent space. The RVQ
         gives a discrete codebook lookup whose geometry moves every step, which
         makes the energy score - a DISTANCE - scale with a target the same gradient
         step is reshaping.
      2. A PER-PATCH POSTERIOR, which the energy score's target draws come from.
         Without one the target is a point and the score degenerates into the
         mean-seeking regression the construction exists to avoid.
      3. ITS OWN RECONSTRUCTION OBJECTIVE, which is what makes the latent
         informative. A bare ``nn.Linear(D, 2*D)`` posterior has only a KL pulling
         it to the prior and a distant byte CE, so nothing requires its latent to
         mean anything.

    So: the same autoencoder at the level this architecture needs it,
    ``h -> (mu, logvar) -> z -> h_hat``, trained on its own relative reconstruction
    error and its own free-bits KL. It shares ``ResidualMLPBlock`` with ``CALMVAE``
    (the reference's ``AELayer`` shape) and the same ``normalize_latent`` contract.

    Running this beside the RVQ is two encoders sharing one trunk, well-posed only
    because the patching is STATIC - both emit exactly one latent per patch, so
    ``z_q + z_c`` is an alignable merge.
    """

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int,
        hidden_dim: int,
        depth: int = 2,
        latent_norm: bool = True,
        dropout: float = 0.15,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.latent_norm = bool(latent_norm)
        self.dropout_p = float(dropout)

        def _drop():
            return nn.Dropout(dropout)

        self.enc_in = nn.Linear(feature_dim, hidden_dim)
        self.enc_blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, _drop(), activation) for _ in range(depth)]
        )
        self.params_norm = nn.RMSNorm(hidden_dim)
        self.to_params = nn.Linear(hidden_dim, 2 * latent_dim)

        self.dec_in = nn.Linear(latent_dim, hidden_dim)
        self.dec_blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, _drop(), activation) for _ in range(depth)]
        )
        self.dec_out = nn.Linear(hidden_dim, feature_dim)
        self.out_norm = nn.RMSNorm(hidden_dim)

    def encode(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``[..., feature_dim]`` -> ``(mean, logvar)``, each ``[..., latent_dim]``.

        Input corruption first, the reference's first dropout site (it zeroes
        random token ids; the continuous analogue is dropping feature
        channels). Forces the latent to DENOISE its patch rather than memorize
        it, which is half of what makes the decoder tolerant of a latent the LM
        predicted rather than encoded.
        """
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        x = self.enc_in(h)
        for blk in self.enc_blocks:
            x = blk(x)
        mean, logvar = self.to_params(self.params_norm(x)).chunk(2, dim=-1)
        return mean, logvar.clamp(min=-10.0, max=10.0)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mean + (0.5 * logvar).exp() * torch.randn_like(mean)

    def normalize_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Unit per-dim RMS. Same contract as ``CALMVAE.normalize_latent``: one
        source of truth for "the decoder consumes a normalized latent", so the
        geometry the energy head predicts into is stationary by construction
        rather than by a correction applied at the loss."""
        if not self.latent_norm:
            return x
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``[..., latent_dim]`` -> ``[..., feature_dim]`` reconstruction.

        LATENT DROPOUT, and it is the whole answer to the train/test gap this
        codec otherwise has. At generation the decoder is handed a latent the
        energy head PREDICTED; in training it would only ever see one the
        encoder produced from real bytes. Dropping the latent teaches the
        decoder to map a NEIGHBOURHOOD of z to the right features, so an
        imperfect prediction still decodes. The reference does exactly this
        (``ae_dropout`` on the sampled latent) and so does ``CALMVAE``, whose
        docstring calls these sites load-bearing for generation.

        Only the RECONSTRUCTION path is perturbed. The trunk consumes the clean
        ``z_c``, and the energy score's target is the clean posterior mean -
        as in the reference, where the LM never sees a dropped latent either.
        """
        z = self.normalize_latent(z)
        z = F.dropout(z, p=self.dropout_p, training=self.training)
        x = self.dec_in(z)
        for blk in self.dec_blocks:
            x = blk(x)
        return self.dec_out(self.out_norm(x))

    def reconstruction_loss(self, h_hat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """RELATIVE squared error: 1.0 is the trivial predict-zero solution.

        Dimensionless on purpose. A raw MSE would carry the scale of whatever
        the local encoder happens to emit, which is the same accident the
        ln(K) code-CE normalization and the per-dimension KL removed - and the
        learned loss balance cannot balance objectives whose units are set by
        unrelated parts of the model.
        """
        return (h_hat - h).pow(2).mean() / h.detach().pow(2).mean().clamp_min(1e-6)

    @staticmethod
    def kl_divergence(
        mean: torch.Tensor, logvar: torch.Tensor, per_dim_clip: float = 0.0
    ) -> torch.Tensor:
        """Per-DIMENSION mean KL against N(0, I), free-bits clipped.

        A mean rather than ``CALMVAE``'s sum: the reference sums over
        latent_size and then applies ``kl_weight=1e-3``, scaling it back down
        by roughly 1/D anyway. Summing without that weight makes the term grow
        with model width - at logvar -8 over D=272 that was 952 nats at step 0,
        which is exactly how abstractinator-p's loss reached 16760.
        """
        per_dim = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar)
        if per_dim_clip and per_dim_clip > 0.0:
            per_dim = per_dim.clamp(min=per_dim_clip)
        return per_dim.mean()
