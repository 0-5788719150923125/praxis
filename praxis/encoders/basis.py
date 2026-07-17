"""Deterministic basis builders shared across encoders.

Extracted from the CALM codec module so non-CALM encoders (the Abstractinator's
harmonic bottleneck) can build the SAME standing-wave coordinate frame the
``HarmonicCodec`` uses - the shared geometry is the point, not a convenience.
Pure math over torch, no praxis imports, so any encoder package can depend on
this without cycles.
"""

import math

import torch


def orthonormal(rows: int, cols: int, seed: int) -> torch.Tensor:
    """Deterministic ``[rows, cols]`` with orthonormal columns (QR of a fixed
    Gaussian). Falls back gracefully when ``rows < cols`` (rank-deficient)."""
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(rows, max(cols, rows), generator=g)[:, :cols]
    q, _ = torch.linalg.qr(m)
    return q[:, :cols]


def harmonic_matrix(rows: int, cols: int) -> torch.Tensor:
    """Orthonormal ``[rows, cols]`` whose columns are the lowest-frequency
    standing waves (DC + cos/sin) over the row index - a structured,
    deterministic alternative to ``orthonormal``. Every column couples all rows
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


def separable_harmonic_matrix(k: int, e: int, cols: int) -> torch.Tensor:
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
    bk = harmonic_matrix(k, k)  # position-frequency modes
    be = harmonic_matrix(e, e)  # feature-frequency modes
    pairs = sorted(
        ((i, j) for i in range(k) for j in range(e)),
        key=lambda p: (p[0] + p[1], p[0], p[1]),
    )[:cols]
    return torch.stack([torch.kron(bk[:, i], be[:, j]) for i, j in pairs], dim=1)
