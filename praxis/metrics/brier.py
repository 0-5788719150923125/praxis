"""BrierLM metric.

Sample-based proper scoring rule from the CALM paper (arXiv 2510.27688,
section 3.5). BrierLM computes per-order Brier scores on n-grams drawn
from the model and the reference text, then returns the geometric mean
over ``n in {1..4}`` scaled by 100.

Typical usage: feed a batch of prefixes to the model, draw two i.i.d.
continuations per prefix, and call ``compute_brier_lm(samples_a,
samples_b, references)``.

Implementation detail: the per-position Brier score only needs pairwise
n-gram matches - no probability access - which makes it model-agnostic
and usable for both CALM and discrete LMs.
"""

from typing import List, Sequence

import torch


def _ngrams(seq: Sequence[int], n: int) -> List[tuple]:
    if len(seq) < n:
        return []
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def _brier_order(
    samples_a: Sequence[Sequence[int]],
    samples_b: Sequence[Sequence[int]],
    references: Sequence[Sequence[int]],
    n: int,
) -> float:
    """Brier-n over a batch of paired samples and a reference.

    For each example:
        B_n = 2 * mean(match(sample_a, reference) at each n-gram position)
            - mean(match(sample_a, sample_b))

    Returned value is the batch mean of ``B_n``. Matches are indicator
    functions over n-gram equality with shared support (fixed ``n``).
    """
    scores = []
    for a, b, r in zip(samples_a, samples_b, references):
        a_ng = _ngrams(a, n)
        b_ng = _ngrams(b, n)
        r_ng = _ngrams(r, n)
        if not a_ng or not r_ng:
            continue
        match_ar = sum(1 for g in a_ng if g in r_ng) / max(len(a_ng), 1)
        match_br = sum(1 for g in b_ng if g in r_ng) / max(len(b_ng), 1)
        match_ab = sum(1 for g in a_ng if g in b_ng) / max(len(a_ng), 1)
        # Symmetrise the positive term so a and b contribute equally.
        scores.append((match_ar + match_br) - match_ab)
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def compute_brier_lm(
    samples_a: Sequence[Sequence[int]],
    samples_b: Sequence[Sequence[int]],
    references: Sequence[Sequence[int]],
    orders: Sequence[int] = (1, 2, 3, 4),
) -> float:
    """Compute BrierLM over a batch.

    Args:
        samples_a, samples_b: paired sample continuations per prompt
            (each a list of token-id lists).
        references: reference continuations per prompt.
        orders: n-gram orders to aggregate (default 1-4, per the paper).

    Returns:
        Geometric mean of per-order Brier scores, scaled by 100. Returns
        0.0 if any order has a non-positive score (the geometric mean is
        ill-defined in that case).
    """
    assert len(samples_a) == len(samples_b) == len(references)
    per_order = []
    for n in orders:
        s = _brier_order(samples_a, samples_b, references, n)
        if s <= 0.0:
            return 0.0
        per_order.append(s)
    # Geometric mean.
    log_mean = sum(torch.tensor(p).log() for p in per_order) / len(per_order)
    return float(log_mean.exp()) * 100.0
