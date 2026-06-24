"""Non-VAE codecs for CALM: drop-in alternatives to ``CALMVAE``.

All share ``CALMVAE``'s constructor signature and ``encode``/``decode``/
``reparameterize``/``normalize_latent``/``kl_divergence`` surface, so the
encoder swaps between them via ``codec_kind`` (see ``CODEC_REGISTRY``). They sit
at different points on the bias-variance codec axis:

  - ``FixedCodec`` (pure bias): a deterministic, non-learned encoder (frozen
    orthonormal byte embedding + orthonormal mix, RMS-normalized) with only the
    decoder learned. The hypothesis: the VAE's two-stage freeze exists only
    because the head needs a *stationary* target and the codec is learned-and-
    moving; a fixed encoder gives a stationary target from step 0, so no freeze
    is needed (pair with ``ae_freeze_steps=0`` for single-stage). The codec is a
    fixed mathematical object, not a learned tokenizer.
  - ``HybridCodec`` (the midpoint): the fixed scaffold plus a small, zero-init,
    gain-bounded learned residual - mostly stationary, but never frozen, so the
    latent can slowly organize over a long run.
  - ``HarmonicCodec``: the fixed scaffold with its *random* orthonormal bases
    replaced by *harmonic* (standing-wave) ones - structured rather than random
    geometry, optionally with a learned periodic ``Serpent`` nonlinearity.

All are lossy by construction (latent_dim < K*embed_dim), so the learned decoder
recovers what it can - the open question being whether a static/structured
latent is "good enough" to model at our tiny scale + 264-byte vocab.
"""

import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from praxis.activations import ACT2CLS
from praxis.activations.serpent import Serpent
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


def _harmonic_matrix(rows: int, cols: int) -> torch.Tensor:
    """Orthonormal ``[rows, cols]`` whose columns are the lowest-frequency
    standing waves (DC + cos/sin) over the row index - a structured,
    deterministic alternative to ``_orthonormal``. Every column couples all rows
    through a shared frequency, so the resulting transform links features via a
    standing wave rather than an arbitrary rotation."""
    idx = torch.arange(rows, dtype=torch.float32)
    feats = [torch.ones(rows)]  # DC
    k = 1
    while len(feats) < cols:
        ang = math.pi * (idx + 0.5) * k / rows
        feats.append(torch.cos(ang))
        if len(feats) < cols:
            feats.append(torch.sin(ang))
        k += 1
    q, _ = torch.linalg.qr(torch.stack(feats[:cols], dim=1))
    return q[:, :cols]


def _separable_harmonic_matrix(k: int, e: int, cols: int) -> torch.Tensor:
    """Separable 2D harmonic basis over (K position, embed feature), flattened
    position-major to ``[k*e, cols]``.

    Columns are 2D standing waves ``kron(B_K[:,i], B_E[:,j])`` for orthonormal
    per-axis harmonic bases, kept in increasing total frequency ``i+j`` so the
    lowest-order modes (smooth across positions AND features) are retained.
    Unlike a 1D harmonic over the flattened index, the K-position axis gets its
    own explicit frequency budget, so smooth-across-patch structure compresses
    into few coefficients as K grows (the large-K mechanism). Columns stay
    orthonormal (Kronecker of orthonormal bases)."""
    cols = min(cols, k * e)
    bk = _harmonic_matrix(k, k)  # position-frequency modes
    be = _harmonic_matrix(e, e)  # feature-frequency modes
    pairs = sorted(
        ((i, j) for i in range(k) for j in range(e)),
        key=lambda p: (p[0] + p[1], p[0], p[1]),
    )[:cols]
    return torch.stack([torch.kron(bk[:, i], be[:, j]) for i, j in pairs], dim=1)


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
        self.activation = activation

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
            [ResidualMLPBlock(hidden_dim, _drop(), activation) for _ in range(depth)]
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


# The learned residual is bounded (tanh-squashed) to this fraction of the fixed
# scaffold, so the latent stays mostly the stationary fixed map while a small
# learned part can slowly bend it. Fixed, model-agnostic.
HYBRID_RESIDUAL_GAIN = 0.1


class HybridCodec(FixedCodec):
    """FixedCodec scaffold + a small, never-frozen learned residual.

    The middle of the bias-variance codec axis: the fixed codec's stationary,
    smooth latent is the dominant term, and a gain-bounded learned MLP adds a
    slow correction on top (driven by reconstruction). Zero-initialised, so the
    codec starts *identical* to FixedCodec and drifts from there - the latent is
    mostly stationary (head still trains fast) but the encoder is never frozen
    and can slowly organize toward a better-conditioned latent over a long run.
    Reclaims the learned-encoder feature without two-stage training or KL.
    Single-stage (pair with ae_freeze_steps=0). The "slowness" is the fixed gain
    bound, not a tuned LR, so it stays inside the no-per-experiment-tuning rule.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual_net = nn.Sequential(
            nn.Linear(self.chunk_size * self.embed_dim, self.hidden_dim),
            ACT2CLS[self.activation](),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        # Zero the output so z starts exactly at the fixed scaffold; the residual
        # grows in from there.
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)

    def encode(self, input_ids: torch.Tensor):
        B, L = input_ids.shape
        K = self.chunk_size
        assert L % K == 0, f"seq len {L} not divisible by chunk size {K}"
        N = L // K
        emb = self.tok_basis[input_ids].view(B, N, K * self.embed_dim)
        z = emb @ self.mix + HYBRID_RESIDUAL_GAIN * torch.tanh(self.residual_net(emb))
        z = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-5)
        return z, self.fixed_logvar.expand_as(z)


class HarmonicCodec(FixedCodec):
    """FixedCodec with its random orthonormal bases replaced by harmonic ones.

    The per-vocab byte embedding becomes a standing-wave signature (a harmonic
    basis over the byte index), and the K*embed -> latent mix becomes a SEPARABLE
    2D harmonic transform over (K position, embed feature) - so the K-position
    axis has its own explicit frequency budget and smooth-across-patch structure
    compresses into few coefficients as K grows (the large-K mechanism the
    scaling conjecture predicts). The latent is a frequency-domain view of the
    patch, every feature coupled through the shared spectrum. Still deterministic
    and stationary like FixedCodec - the geometry is structured, not random.

    ``nonlinear=True`` applies a learned periodic ``Serpent`` activation after the
    transform - a gentle (residual ``x + oscillation``) nonlinearity that keeps
    most of the smooth linear scaffold while adding harmonic curvature. This makes
    the encode learnable (single-stage, never frozen), unlike the linear default.

    Caveat: the per-vocab harmonic embedding imposes a weak ORDINAL prior on byte
    ids (nearby byte values -> similar signatures), only loosely true for a byte
    alphabet. If it hurts, keep tok_basis random and harmonic-ize only the mix.
    """

    def __init__(self, *args, nonlinear: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tok_basis.copy_(
            F.normalize(_harmonic_matrix(self.vocab_size, self.embed_dim), dim=-1)
        )
        self.mix.copy_(
            _separable_harmonic_matrix(
                self.chunk_size, self.embed_dim, self.latent_dim
            )
        )
        self.act = Serpent() if nonlinear else None

    def encode(self, input_ids: torch.Tensor):
        B, L = input_ids.shape
        K = self.chunk_size
        assert L % K == 0, f"seq len {L} not divisible by chunk size {K}"
        N = L // K
        emb = self.tok_basis[input_ids].view(B, N, K * self.embed_dim)
        z = emb @ self.mix
        if self.act is not None:
            z = self.act(z)
        z = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-5)
        return z, self.fixed_logvar.expand_as(z)


# Codec slot, selected by the encoder's codec_kind kwarg (profile partial).
from praxis.encoders.calm.vae import CALMVAE  # noqa: E402

CODEC_REGISTRY = {
    "vae": CALMVAE,
    "fixed": FixedCodec,
    "hybrid": HybridCodec,
    "harmonic": HarmonicCodec,
    "harmonic_serpent": partial(HarmonicCodec, nonlinear=True),
}
