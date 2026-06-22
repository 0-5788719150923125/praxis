"""Fixed (non-learned) codec for CALM: a drop-in alternative to ``CALMVAE``.

The hypothesis (worth a shot at our tiny scale + 264-byte vocab): the VAE's
two-stage freeze exists only because the AR head needs a *stationary* latent
target, and the codec is learned-and-moving. So fix the **encoder** as a pure
deterministic function ``bytes -> z`` (no parameters, stationary from step 0),
and keep only the **decoder** learned. Then:

  - the AR head's target never moves, so no stage-1 freeze is needed (pair this
    with ``ae_freeze_steps=0`` for single-stage joint training);
  - the decoder trains continuously and is never frozen - it just learns to
    invert the fixed map, and improving it can't move the head's target;
  - the "codec" is a fixed mathematical object, not a learned tokenizer.

The fixed encoder is a frozen orthonormal byte embedding followed by a frozen
orthonormal mix of the K concatenated embeddings, RMS-normalized to a clean
unit-scale latent. Lossy by construction (latent_dim < K*embed_dim), so the
learned decoder recovers what it can - exactly the open question this ablation
asks: is a static latent "good enough" to model at this scale.

Same constructor signature and ``encode``/``decode``/``reparameterize``/
``normalize_latent``/``kl_divergence`` surface as ``CALMVAE`` for registry
interchange (see ``CODEC_REGISTRY``).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from praxis.encoders.calm.vae import HarmonicDropout, ResidualMLPBlock

# Deterministic build seed for the frozen bases (reproducible, resume-stable).
FIXED_CODEC_SEED = 1234
# Fixed posterior std: a tight ball around the deterministic mean so the head
# has a near-point target without a degenerate (zero-variance) one.
FIXED_CODEC_STD = 0.1


def _orthonormal(rows: int, cols: int, seed: int) -> torch.Tensor:
    """Deterministic ``[rows, cols]`` with orthonormal columns (QR of a fixed
    Gaussian). Falls back gracefully when ``rows < cols`` (rank-deficient)."""
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(rows, max(cols, rows), generator=g)[:, :cols]
    q, _ = torch.linalg.qr(m)
    return q[:, :cols]


class FixedCodec(nn.Module):
    """Fixed deterministic encoder + learned decoder. Drop-in for ``CALMVAE``."""

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

        # Frozen encode path (buffers, never learned). Unit-norm byte embeddings
        # and an orthonormal K*embed -> latent mix; z is a pure function of ids.
        tok = _orthonormal(vocab_size, embed_dim, FIXED_CODEC_SEED)
        tok = F.normalize(tok, dim=-1)
        self.register_buffer("tok_basis", tok, persistent=True)  # [V, E]
        mix = _orthonormal(chunk_size * embed_dim, latent_dim, FIXED_CODEC_SEED + 1)
        self.register_buffer("mix", mix, persistent=True)  # [K*E, L]
        self.register_buffer(
            "fixed_logvar", torch.tensor(2.0 * math.log(FIXED_CODEC_STD))
        )

        def _drop():
            if dropout_mode == "harmonic":
                return HarmonicDropout(dropout, n_cycles=dropout_cycles)
            return nn.Dropout(dropout)

        # Learned decoder (the only trainable part): latent -> K per-token feats.
        self.dec_in = nn.Linear(latent_dim, hidden_dim)
        self.dec_blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, _drop()) for _ in range(depth)]
        )
        self.dec_expand = nn.Linear(hidden_dim, chunk_size * hidden_dim)
        self.out_norm = nn.RMSNorm(hidden_dim)

    def encode(self, input_ids: torch.Tensor):
        """Deterministic ``[B, N*K]`` ids -> ``(mean, logvar)`` ``[B, N, L]``.

        No input dropout (the target must be a clean function of the input);
        the mean is RMS-normalized so the fixed latent is well-conditioned, and
        logvar is a fixed constant.
        """
        B, L = input_ids.shape
        K = self.chunk_size
        assert L % K == 0, f"seq len {L} not divisible by chunk size {K}"
        N = L // K
        emb = self.tok_basis[input_ids]  # [B, N*K, E]
        emb = emb.view(B, N, K * self.embed_dim)
        z = emb @ self.mix  # [B, N, L]
        z = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-5)
        return z, self.fixed_logvar.expand_as(z)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mean + std * torch.randn_like(std)

    def normalize_latent(self, x: torch.Tensor) -> torch.Tensor:
        """No-op: ``encode`` already RMS-normalizes the latent. Present for
        interface parity (decode and the head's target both call it)."""
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Learned latent -> ``[B, N*K, hidden_dim]`` decoder features (same
        layout as ``CALMVAE.decode``)."""
        B, N, _ = z.shape
        K = self.chunk_size
        z = F.dropout(z, p=self.dropout_p, training=self.training)
        h = self.dec_in(z)
        for blk in self.dec_blocks:
            h = blk(h)
        h = self.dec_expand(h).view(B, N, K, self.hidden_dim)
        h = h.reshape(B, N * K, self.hidden_dim)
        return self.out_norm(h)

    @staticmethod
    def kl_divergence(mean, logvar, per_dim_clip: float = 0.0) -> torch.Tensor:
        """Zero: the encoder is fixed, so there is no posterior to regularize.
        Stage-1 loss is then pure reconstruction (decoder training)."""
        return mean.new_zeros(mean.shape[:-1])


# Codec slot, selected by the encoder's codec_kind kwarg (profile partial).
from praxis.encoders.calm.vae import CALMVAE  # noqa: E402

CODEC_REGISTRY = {
    "vae": CALMVAE,
    "fixed": FixedCodec,
}
