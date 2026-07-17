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
from torch import nn

from praxis.activations.serpent import Serpent
from praxis.encoders.basis import harmonic_matrix

from .vector_quantizer import MultiStageResidualVQ


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
        self.quantizer = MultiStageResidualVQ(D=latent_dim, **vq_kwargs)

    def forward(self, h: torch.Tensor):
        z = h @ self.analysis  # [B, N, L] harmonic amplitudes
        if self.act is not None:
            z = self.act(z)
        # Spectral energy normalization: quantize direction on the sphere.
        z = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-5)
        z_q, vq_loss, indices, perplexity = self.quantizer(z)
        return z_q @ self.analysis.T, vq_loss, indices, perplexity
