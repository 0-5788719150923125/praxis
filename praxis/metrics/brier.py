"""BrierLM metric.

Sample-based proper scoring rule from the CALM paper (arXiv 2510.27688,
section 3.5). For each n-gram order the per-position Brier score is the
likelihood-free estimator

    1{x1 = y} + 1{x2 = y} - 1{x1 = x2}

where x1, x2 are two i.i.d. sampled continuations and y is the reference.
Matches are *position-aligned exact prefix matches*: order n counts a hit
only when the next n tokens agree on every position, in order (the
reference's ``cumprod`` over consecutive equality indicators). BrierLM is
the geometric mean over n in {1..4}.

This mirrors the reference ``eval_brier`` / ``compute_metrics``
(github.com/shaochenze/calm). Samples and reference must be aligned,
equal-length token-id sequences (the model's continuation against the true
continuation of the same prompt).
"""

from typing import List, Optional, Sequence

import torch

# The reference omits this; we scale for chart readability (val_brierlm).
BRIERLM_SCALE = 100.0


def _aligned_eq(x: Sequence[int], y: Sequence[int]) -> torch.Tensor:
    """Per-position equality indicators for two aligned id sequences."""
    L = min(len(x), len(y))
    return torch.tensor(
        [1.0 if x[i] == y[i] else 0.0 for i in range(L)], dtype=torch.float32
    )


def _prefix_match_rate(eq: torch.Tensor, n: int) -> Optional[float]:
    """Fraction of length-n windows that match on every position.

    A window is a hit iff the running product over its n equality indicators
    is 1 - the reference's ``cumprod`` semantics, slid across all positions.
    """
    if eq.numel() < n:
        return None
    windows = eq.unfold(0, n, 1)  # [num_windows, n]
    return float(windows.prod(dim=1).mean())


def _brier_order(
    samples_a: Sequence[Sequence[int]],
    samples_b: Sequence[Sequence[int]],
    references: Sequence[Sequence[int]],
    n: int,
) -> Optional[float]:
    """Brier-n: batch mean of ``1{a=y} + 1{b=y} - 1{a=b}`` over aligned
    length-n prefix matches. None if no example is long enough for order n."""
    scores = []
    for a, b, r in zip(samples_a, samples_b, references):
        ar = _prefix_match_rate(_aligned_eq(a, r), n)
        if ar is None:
            continue
        br = _prefix_match_rate(_aligned_eq(b, r), n)
        ab = _prefix_match_rate(_aligned_eq(a, b), n)
        scores.append(ar + br - ab)
    if not scores:
        return None
    return sum(scores) / len(scores)


def compute_brier_lm(
    samples_a: Sequence[Sequence[int]],
    samples_b: Sequence[Sequence[int]],
    references: Sequence[Sequence[int]],
    orders: Sequence[int] = (1, 2, 3, 4),
) -> float:
    """Compute BrierLM over a batch.

    Args:
        samples_a, samples_b: paired sampled continuations per prompt, each a
            list of token-id lists, aligned to ``references``.
        references: the true continuation per prompt.
        orders: n-gram orders to aggregate (default 1-4, per the paper).

    Returns:
        ``max(prod(brier_n), 0) ** (1/k)`` scaled by ``BRIERLM_SCALE``. Each
        order is floored at 0 first: a non-positive order is no positive
        evidence, so it zeroes the geometric mean rather than flipping its
        sign in the product.
    """
    assert len(samples_a) == len(samples_b) == len(references)
    prod = 1.0
    count = 0
    for n in orders:
        s = _brier_order(samples_a, samples_b, references, n)
        if s is None:
            continue
        prod *= max(s, 0.0)
        count += 1
    if count == 0 or prod <= 0.0:
        return 0.0
    return (prod ** (1.0 / count)) * BRIERLM_SCALE
