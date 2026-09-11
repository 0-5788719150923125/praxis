"""BrierLM (praxis.metrics.compute_brier_lm): 100 for exact agreement, 0 for none."""

import pytest

from praxis.metrics import compute_brier_lm


def test_brierlm_scores_range():
    refs = [[1, 2, 3, 4, 5, 6, 7, 8] for _ in range(4)]
    a_same = [list(r) for r in refs]
    b_same = [list(r) for r in refs]
    assert compute_brier_lm(a_same, b_same, refs) == pytest.approx(100.0)

    a_rand = [[999] * 8 for _ in range(4)]
    b_rand = [[999] * 8 for _ in range(4)]
    assert compute_brier_lm(a_rand, b_rand, refs) == 0.0
