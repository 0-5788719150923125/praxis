"""Residual VQ in a fixed harmonic coordinate frame.

The bridge the paper's Abstractinator addendum states as conjecture
(praxis/pillars/framing/harmonic-codec-abstractinator.yml): residual codes
read as amplitudes in the harmonic basis. Patch latents are rotated into the
same standing-wave basis the CALM ``HarmonicCodec`` builds its mix from
(praxis/encoders/basis.py), RMS-normalized there (spectral energy
normalization, so the latent lives on a sphere in coefficient space like the
codec's normalized latent), residual-quantized by the existing
``MultiStageResidualVQ``, and synthesized back through the adjoint. Each code
is then a coarse-to-fine address over harmonic amplitudes rather than raw
feature coordinates.

``nonlinear=True`` adds a learned periodic ``Serpent`` activation after the
analysis transform, mirroring ``codec_kind="harmonic_serpent"`` on the CALM
side: the encode into the spectral frame becomes learnable (still single-stage
and never frozen) instead of a fixed rotation.

Unlike the CALM codecs there is no freeze, no KL, and no latent prediction
target - the quantizer's loss folds into the encoder's aux_loss and the model
trains end-to-end on byte cross-entropy.
"""

import torch
import torch.nn.functional as F
from torch import nn

from praxis.activations.serpent import Serpent
from praxis.encoders.basis import harmonic_matrix

from .vector_quantizer import MultiStageResidualVQ


class GDN(nn.Module):
    """Generalized divisive normalization (Balle et al., arXiv:1611.01704).

    ``y_i = x_i / sqrt(beta_i + sum_j gamma_ij * x_j^2)``

    A learned compander for the quantizer to sit behind. Classical companding
    (mu-law and friends) exists because a fixed codebook should spend its
    resolution where the source density actually is; a monotonic warp reshapes
    the density so a uniform quantizer covers it well. GDN is the learned,
    multivariate version, and it is what neural image codecs put in front of
    their quantizer for exactly this reason.

    The RMS normalization it replaces is the ISOTROPIC SPECIAL CASE of this:
    ``gamma_ij = 1/L`` for all (i, j) and ``beta = 1e-5`` reduces the expression
    to ``x * rsqrt(mean(x^2) + 1e-5)``. So the defaults here initialize to
    exactly the previous behaviour, bit for bit, and the run measures whether
    letting the frame depart from isotropy earns anything.

    Positivity of beta and gamma is enforced by squaring an unconstrained
    parameter, as in the reference implementation - a projection/clamp puts a
    kink in the gradient right where the normalizer is most sensitive.
    """

    def __init__(self, dim: int, beta_min: float = 1e-6) -> None:
        super().__init__()
        self.beta_min = beta_min
        # beta^2 + beta_min == 1e-5 at init, matching the old epsilon.
        self.beta_sqrt = nn.Parameter(torch.full((dim,), (1e-5 - beta_min) ** 0.5))
        # gamma_ij = 1/dim at init -> the sum below IS the mean over features.
        self.gamma_sqrt = nn.Parameter(torch.full((dim, dim), (1.0 / dim) ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = self.beta_sqrt.pow(2) + self.beta_min
        gamma = self.gamma_sqrt.pow(2)
        norm = torch.sqrt(F.linear(x.pow(2), gamma) + beta)
        return x / norm


class HarmonicResidualVQ(nn.Module):
    """Drop-in for ``MultiStageResidualVQ`` with harmonic coordinates.

    Same forward contract: ``[B, N, D] -> (z_q, vq_loss, indices, perplexity)``
    with ``z_q`` back in the model's ``D``-dimensional space. ``latent_dim < D``
    makes the frame lossy (a low-frequency spectral budget, the codec's
    ``latent_dim < K*embed_dim`` mechanism); ``latent_dim == D`` is a pure
    rotation and the bottleneck is the quantizer alone.
    """

    def __init__(
        self,
        dim: int,
        latent_dim: int,
        nonlinear: bool = False,
        normalization: str = "rms",
        **vq_kwargs,
    ) -> None:
        super().__init__()
        latent_dim = max(1, min(latent_dim, dim))
        # Fixed analysis frame [D, L]; synthesis is its adjoint (orthonormal
        # columns), exact on the retained subspace. Deterministic, so persistence
        # is for resume parity with the CALM codec buffers, not necessity.
        self.register_buffer(
            "analysis", harmonic_matrix(dim, latent_dim), persistent=True
        )
        self.act = Serpent() if nonlinear else None
        if normalization not in ("rms", "gdn"):
            raise ValueError(f"Unknown bottleneck normalization: {normalization!r}")
        self.gdn = GDN(latent_dim) if normalization == "gdn" else None
        self.quantizer = MultiStageResidualVQ(D=latent_dim, **vq_kwargs)

    def forward(self, h: torch.Tensor):
        z = h @ self.analysis  # [B, N, L] harmonic amplitudes
        if self.act is not None:
            z = self.act(z)
        if self.gdn is not None:
            # Learned compander. Initializes to the rsqrt-of-mean below exactly,
            # then is free to allocate resolution anisotropically.
            z = self.gdn(z)
        else:
            # Spectral energy normalization: quantize direction on the sphere.
            z = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-5)
        z_q, vq_loss, indices, perplexity = self.quantizer(z)
        return z_q @ self.analysis.T, vq_loss, indices, perplexity
