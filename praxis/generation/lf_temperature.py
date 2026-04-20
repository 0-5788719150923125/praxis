"""Locally-Fair (LF) temperature sampling for continuous latents.

The CALM energy head defines an implicit distribution over latents
``z`` given a conditioning ``h``. Because the density is not tractable,
standard temperature-scaled softmax sampling does not apply.

Algorithms 1 and 2 below follow the paper (arXiv 2510.27688, section
3.4). Both use importance weights built from a kernel density estimate
over a batch of samples drawn from the base distribution.

- Algorithm 1 (exact): rejection sampling. The proposal is the base
  distribution; the target has density proportional to ``f(z)^((1-T)/T)``
  where ``f`` is a KDE over an auxiliary batch of size ``S``. Accept
  rate drops with extreme temperatures; we cap retries.

- Algorithm 2 (approximate): draw ``S`` samples once, weight each by
  ``f(z_i)^((1-T)/T)`` (normalized), and pick one via categorical
  sampling. Much cheaper, and the standard path unless extreme fidelity
  to Algorithm 1 is needed.

Both reduce to a plain draw from the base distribution at ``T=1``.
"""

from typing import Callable

import torch


def _kde_weights(
    z: torch.Tensor, bandwidth: float = 1.0, eps: float = 1e-12
) -> torch.Tensor:
    """Gaussian KDE log-density estimate at each row of ``z``.

    ``z``: ``[S, D]``. Returns ``[S]`` of (log-)density estimates.
    """
    S, D = z.shape
    # Pairwise squared distances, excluding the self-pair.
    diff = z.unsqueeze(0) - z.unsqueeze(1)  # [S, S, D]
    sq = diff.pow(2).sum(dim=-1)  # [S, S]
    mask = 1.0 - torch.eye(S, device=z.device, dtype=z.dtype)
    # Median heuristic for bandwidth if not set.
    if bandwidth is None or bandwidth <= 0:
        with torch.no_grad():
            med = sq[mask.bool()].median().clamp_min(eps)
            bandwidth = float(med.sqrt().item()) + eps
    kernels = torch.exp(-sq / (2.0 * bandwidth * bandwidth)) * mask
    density = kernels.sum(dim=-1) / max(S - 1, 1)
    return torch.log(density.clamp_min(eps))


def lf_temperature_sample_batched(
    base_sampler: Callable[[int], torch.Tensor],
    temperature: float,
    num_candidates: int = 64,
    bandwidth: float = 1.0,
) -> torch.Tensor:
    """Algorithm 2: batched approximate LF-temperature sampling.

    Args:
        base_sampler: Callable ``n -> [n, D]`` that draws ``n`` i.i.d.
            samples from the base distribution (the energy head at a
            fixed conditioning ``h``).
        temperature: ``T``. ``T == 1`` short-circuits to a single draw.
        num_candidates: ``S``. Number of candidates for the KDE.
        bandwidth: Gaussian kernel bandwidth (in latent-space units).

    Returns:
        ``[D]`` selected latent.
    """
    if temperature == 1.0 or num_candidates <= 1:
        return base_sampler(1).squeeze(0)

    z = base_sampler(num_candidates)  # [S, D]
    log_f = _kde_weights(z, bandwidth=bandwidth)
    alpha = (1.0 - temperature) / temperature
    logits = alpha * log_f
    logits = logits - logits.max()
    probs = torch.softmax(logits, dim=0)
    idx = torch.multinomial(probs, 1)
    return z[idx].squeeze(0)


def lf_temperature_sample_exact(
    base_sampler: Callable[[int], torch.Tensor],
    temperature: float,
    num_candidates: int = 64,
    bandwidth: float = 1.0,
    max_tries: int = 256,
) -> torch.Tensor:
    """Algorithm 1: rejection-sampled LF-temperature sampling.

    The proposal is the base distribution; the target has density
    proportional to ``f(z)^((1-T)/T)`` with ``f`` a KDE from ``S``
    auxiliary samples. Falls back to the batched estimator if acceptance
    has not succeeded within ``max_tries``.
    """
    if temperature == 1.0:
        return base_sampler(1).squeeze(0)

    z_aux = base_sampler(num_candidates)
    log_f_aux = _kde_weights(z_aux, bandwidth=bandwidth)
    log_ceiling = log_f_aux.max()
    alpha = (1.0 - temperature) / temperature

    for _ in range(max_tries):
        z = base_sampler(1)
        # Density at new z: mean kernel against the auxiliary batch.
        diff = z - z_aux  # [S, D]
        sq = diff.pow(2).sum(dim=-1)
        log_f = torch.log(
            (torch.exp(-sq / (2.0 * bandwidth * bandwidth)).mean()).clamp_min(1e-12)
        )
        log_accept = alpha * (log_f - log_ceiling)
        log_accept = torch.clamp(log_accept, max=0.0)
        if torch.rand(1, device=z.device).log() < log_accept:
            return z.squeeze(0)

    return lf_temperature_sample_batched(
        base_sampler, temperature, num_candidates=num_candidates, bandwidth=bandwidth
    )
