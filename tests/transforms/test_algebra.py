"""Ghost features: the algebra, the in-place parametrization, and the walker.

The properties pinned here are the ones the experiment's interpretation rests on.
If the complex product is wrong the arm is not testing the paper's mechanism; if
the init does not inherit the host module's scale the arms differ by an init as
well as a mechanism; if the walker misses a target the run is not the experiment.
"""

import pytest
import torch

from praxis.transforms.algebra import (
    ALGEBRAS,
    has_antipodal_pair,
    non_degenerate,
    structure_matrices,
    tables,
)

# --- the algebra ------------------------------------------------------------


@pytest.mark.parametrize("algebra", sorted(ALGEBRAS))
def test_algebra_is_non_degenerate(algebra):
    """The paper's own condition: every P_k non-singular (section 2.1). It is what
    carries the universal-approximation guarantee to ANY dimension d, which is
    why d=2 is admissible and why the cyclic family can be any size at all."""
    perm, sign = tables(algebra)
    assert non_degenerate(perm, sign)
    dets = torch.linalg.det(structure_matrices(perm, sign))
    assert torch.allclose(dets.abs(), torch.ones_like(dets))


@pytest.mark.parametrize("algebra", sorted(ALGEBRAS))
def test_algebra_has_no_antipodal_pair(algebra):
    """Stricter than the paper, and earned: this stack runs periodic (odd)
    activations, which commute with a global sign flip, so two blocks related by
    P_j = -P_k would collapse after the nonlinearity."""
    perm, sign = tables(algebra)
    assert not has_antipodal_pair(perm, sign)


def test_cyclic_algebra_is_the_group_algebra_of_z_mod_d():
    """(x.w)_k = sum_{i+j=k mod d} x_i w_j, so PERM[k][p] = (k-p) mod d, signs +1.
    Permutation matrices, so non-degenerate at every d - which is what lets d be
    chosen to DIVIDE THE TENSOR rather than the tensor chosen to suit d."""
    perm, sign = tables("cyclic3")
    assert torch.all(sign == 1)
    for k in range(3):
        for p_ in range(3):
            assert perm[k, p_].item() == (k - p_) % 3
