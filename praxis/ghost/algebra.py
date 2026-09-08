"""Signed-permutation structure matrices for the hypercomplex algebras we use.

The whole of a hypercomplex product, for the algebras that matter here, is an
index-gather plus a sign flip. That is the observation this package is built on,
and it is why nothing in it needs a quaternion library or a new dependency.

For an algebra of dimension ``d``, the product ``x . w`` has ``d`` components,
and component ``k`` is a signed sum over the ``d`` components of ``w``::

    component_k = sum_p  x_p * SIGN[k][p] * w[PERM[k][p]]

so ``PERM[k]`` says which component of ``w`` position ``p`` reads, and
``SIGN[k][p]`` says with what sign. Equivalently ``P_k[p, PERM[k][p]] =
SIGN[k][p]``, which is the matrix form the paper writes the multiplication table
in (Vieira Neto & Valle 2026, arXiv:2608.07735, section 2.1).

THE ONE GUARANTEE WE INHERIT, and it is the reason ``d`` is free. The paper:

    "a hypercomplex algebra is called non-degenerate ... if the matrices
    P_0, ..., P_{d-1} are all non-singular. Non-degenerate hypercomplex algebras
    guarantee the universal approximation capability of hypercomplex-valued
    neural networks and are essential for extracting ghost features."

and, in section 2.2, "the approximation capability holds, in particular, for
complex-, quaternion-, and Clifford-valued MLP networks". Section 3 opens "Let H
be a hypercomplex algebra with dimension d". No result in the paper is specific
to ``d = 4``, and complex numbers are named as covered. A signed permutation
matrix has determinant +-1, so ``non_degenerate()`` below can only fail on a
typo - which is exactly what it is there to catch.

WHAT WE DO NOT INHERIT. The paper's "the real part replicates the original
layer's output" property assumes a SPLIT activation, applied component-wise. Our
first target site feeds a GLU (``a * act(b)``), which mixes two component blocks
multiplicatively. That property exists for their frozen-backbone transfer-
learning setup; we train from scratch and do not need it, so it is not claimed.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

# (PERM, SIGN) per algebra. Row k is output component k.
#
# Complex (d = 2):
#     Re{x.w} = x0 w0 - x1 w1
#     Im{x.w} = x0 w1 + x1 w0
#
# Quaternion (d = 4):
#     Re    = x0 w0 - x1 w1 - x2 w2 - x3 w3
#     Im_1  = x0 w1 + x1 w0 + x2 w3 - x3 w2
#     Im_2  = x0 w2 - x1 w3 + x2 w0 + x3 w1
#     Im_3  = x0 w3 + x1 w2 - x2 w1 + x3 w0
ALGEBRAS: Dict[str, Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]] = {
    "complex": (
        ((0, 1), (1, 0)),
        ((1, -1), (1, 1)),
    ),
    "quaternion": (
        ((0, 1, 2, 3), (1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0)),
        ((1, -1, -1, -1), (1, 1, 1, -1), (1, -1, 1, 1), (1, 1, -1, 1)),
    ),
}


def tables(algebra: str) -> Tuple[Tensor, Tensor]:
    """``(PERM, SIGN)`` as ``[d, d]`` long / float tensors."""
    if algebra not in ALGEBRAS:
        raise ValueError(f"Unknown algebra {algebra!r}; known: {sorted(ALGEBRAS)}")
    perm, sign = ALGEBRAS[algebra]
    return (
        torch.tensor(perm, dtype=torch.long),
        torch.tensor(sign, dtype=torch.float32),
    )


def structure_matrices(perm: Tensor, sign: Tensor) -> Tensor:
    """``[d, d, d]`` stack of ``P_k``, for the non-degeneracy check."""
    d = perm.size(0)
    p = torch.zeros(d, d, d, dtype=torch.float64)
    rows = torch.arange(d)
    for k in range(d):
        p[k, rows, perm[k]] = sign[k].double()
    return p


def non_degenerate(perm: Tensor, sign: Tensor) -> bool:
    """Every ``P_k`` non-singular: the paper's own condition on the algebra."""
    dets = torch.linalg.det(structure_matrices(perm, sign))
    return bool(torch.all(dets.abs() > 1e-9))


def has_antipodal_pair(perm: Tensor, sign: Tensor) -> bool:
    """True if two blocks are a GLOBAL negation of each other.

    Not the paper's condition, and stricter than it. An ODD activation commutes
    with a global sign flip, so two blocks related by ``P_j = -P_k`` would
    collapse to a sign flip after the nonlinearity and cost a full degree of
    freedom. This stack runs periodic activations (``servant``; ``sin`` is odd),
    so the check is earned rather than inherited. Neither ``complex`` nor
    ``quaternion`` has such a pair; any new algebra must be checked before use.
    """
    d = perm.size(0)
    for j in range(d):
        for k in range(j + 1, d):
            if torch.equal(perm[j], perm[k]) and torch.equal(sign[j], -sign[k]):
                return True
    return False
