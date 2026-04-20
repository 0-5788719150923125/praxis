"""Energy-score loss for CALM.

Given N "model" samples and M "target" posterior samples per position,
the energy score is the strictly proper scoring rule

    L = (2 / NM) * sum_{n,m} ||z_m - z_hat_n||
      -  (1 / (N*(N-1))) * sum_{n != k} ||z_hat_n - z_hat_k||

Minimising L drives the distribution over ``z_hat`` to match the
distribution over ``z`` (see section 3.3 of arXiv 2510.27688).

Stop-gradient on the target posterior samples prevents the energy path
from pulling the VAE's posterior toward whatever the LM currently
predicts.
"""

import torch


def _pairwise_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Euclidean distances between rows of ``a`` and rows of ``b``.

    ``a``: ``[..., N, D]``.
    ``b``: ``[..., M, D]``.
    Returns: ``[..., N, M]``.

    ``eps`` floors the squared distance so that ``d/dx sqrt(x)`` stays
    bounded when samples nearly coincide. With ``eps=1e-4`` the worst-case
    grad is ~50 instead of ~5e3 at ``eps=1e-8``.
    """
    diff = a.unsqueeze(-2) - b.unsqueeze(-3)
    return torch.sqrt(diff.pow(2).sum(dim=-1).clamp_min(eps))


def energy_score_loss(
    model_samples: torch.Tensor,
    target_samples: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compute the energy score.

    Args:
        model_samples: ``[..., N, D]`` generator draws ``z_hat``.
        target_samples: ``[..., M, D]`` posterior draws ``z`` - should
            already be detached by the caller.
        eps: Small constant inside the norm to avoid sqrt(0) gradients.

    Returns:
        Scalar (mean over leading dims) energy score.
    """
    # Float32 keeps the repulsive second term stable for small D.
    m = model_samples.float()
    t = target_samples.float()

    N = m.shape[-2]

    attr = _pairwise_distance(m, t, eps=eps).mean(dim=(-2, -1))

    if N > 1:
        self_dists = _pairwise_distance(m, m, eps=eps)
        # Off-diagonal mean; excludes the zero self-pairs.
        mask_eye = torch.eye(N, device=m.device, dtype=m.dtype)
        repulse = (self_dists * (1.0 - mask_eye)).sum(dim=(-2, -1)) / (N * (N - 1))
    else:
        repulse = torch.zeros_like(attr)

    return (2.0 * attr - repulse).mean()
