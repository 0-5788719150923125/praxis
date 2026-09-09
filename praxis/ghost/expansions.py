"""Expansion rules: how a small REAL tensor becomes a full-shape weight.

Every rule owns its real parameters and returns a weight of the shape the module
it replaced had, so the host module's forward is unchanged and nothing
downstream sees a different shape. That is what lets the same wrapper serve the
hypothesis and all of its controls: the arms of the experiment are entries in
``EXPANSION_REGISTRY``, not separate modules or separate integration sites.

    algebra     d blocks that are fixed signed permutations of one real tensor.
                The hypothesis (Vieira Neto & Valle 2026, arXiv:2608.07735).
    random      identical shape and parameter count, arbitrary FROZEN signed
                permutation instead of the algebra's. Not an ablation - the
                actual test of whether the algebra carries anything. If
                ``algebra ~= random``, the mechanism is "fixed expansion plus a
                learned mixer" and the algebra is decoration.
    lowrank     a plain rank-r factorization at the same parameter budget.
                **NOT a matched control - see the warning below.** Kept as a
                reference point for what the obvious alternative costs.

WHY ``lowrank`` IS NOT THE CONTROL IT LOOKS LIKE, measured rather than argued.
Matching the parameter budget on a ``[544, 816]`` unfolded conv weight forces
``r = 221952 / (544 + 816) = 163``, and ``163 < 544``, so the factorization is
RANK-CAPPED at 163 of 544 while every algebra rule produces a FULL-RANK 544:

    complex / random   221,952 params   rank 544
    quaternion         110,976 params   rank 544
    lowrank            221,680 params   rank 163
    lowrank4           111,520 params   rank  82

That is a 3.3x harsher constraint at the same budget, so an arm built on it loses
for a reason unrelated to fixed-versus-learned structure. Low-rank is only a
competitive parameterization when the target matrix is itself nearly low-rank,
and a dilated conv filter bank is not. The first -r run was retired on this.

AND IT SHOWS WHAT THE ALGEBRA IS ACTUALLY BUYING. ``P_k`` acts on the INPUT
axis, not the output axis, so the ``d`` expanded blocks are NOT linear
combinations of the ``out // d`` real rows - each block applies a different
input-space transform, so the rows land in different coordinate arrangements and
the stack is full rank. Output-axis mixing (which is what a low-rank factor is)
cannot do that at any budget. Input-axis structure buys output rank; that is the
mechanism, and it is why the honest control is another SIGNED PERMUTATION
(``random``) rather than another compression scheme.

Rank for ``lowrank`` is DERIVED from the budget the algebra rule spends, not
tuned, so the two are parameter-matched even though they are not rank-matched.
"""

from __future__ import annotations

import math
import zlib
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from praxis.ghost.algebra import (
    has_antipodal_pair,
    non_degenerate,
    structure_matrices,
    tables,
)


def _stable_seed(tag: str) -> int:
    """Reproducible across processes, unlike ``hash()``, which is salted."""
    return zlib.crc32(tag.encode("utf-8")) & 0x7FFFFFFF


class Expansion(nn.Module):
    """Owns the real parameters; ``forward()`` returns the full-shape weight.

    ``shape`` is the ORIGINAL weight shape, ``[out, in, *tail]`` for both
    ``nn.Linear`` (empty tail) and ``nn.Conv1d`` (tail ``(kernel,)``).

    EVERY RULE MUST INIT TO THE SAME SCALE. The expanded weight has to start
    where the module it replaced would have, or the arms differ by an init as
    well as by a mechanism and the comparison measures both at once.
    """

    # torch's default for Linear/Conv weights is `kaiming_uniform_(a=sqrt(5))`,
    # whose gain is `sqrt(2 / (1 + a^2)) = sqrt(1/3)`. Named here because two
    # rules have to reproduce that target rather than inherit it.
    GAIN_SQ: float = 1.0 / 3.0

    def __init__(self, shape: Sequence[int], tag: str = "") -> None:
        super().__init__()
        self.shape: Tuple[int, ...] = tuple(int(s) for s in shape)
        self.tag = tag

    @property
    def real_numel(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def full_numel(self) -> int:
        return math.prod(self.shape)

    def extra_repr(self) -> str:
        return (
            f"{tuple(self.shape)} from {self.real_numel:,} real "
            f"({self.real_numel / max(1, self.full_numel):.3f}x)"
        )


class AlgebraExpansion(Expansion):
    """``d`` output blocks, each a signed permutation of one real tensor.

    The real tensor is ``[out // d, in, *tail]``; block ``k`` reads the INPUT
    axis in groups of ``d`` under ``(PERM[k], SIGN[k])``, and the blocks are
    concatenated along the output axis. Both axes must divide by ``d``: the
    output because that is the halving, the input because that is the grouping.

    INIT. ``fan_in`` is ``in * prod(tail)``, which the real tensor and the
    original share, so the module's own default init applied to the real tensor
    already gives the expanded weight the original's element distribution.
    Nothing here has to renormalize, and no scale factor is being chosen.
    """

    def __init__(
        self,
        shape: Sequence[int],
        algebra: str = "complex",
        tag: str = "",
        randomize: bool = False,
    ) -> None:
        super().__init__(shape, tag)
        perm, sign = tables(algebra)
        self.d = int(perm.size(0))
        self.algebra = f"random_d{self.d}" if randomize else algebra

        if randomize:
            g = torch.Generator().manual_seed(_stable_seed(f"{tag}|{algebra}"))
            perm = torch.stack(
                [torch.randperm(self.d, generator=g) for _ in range(self.d)]
            )
            sign = torch.where(torch.rand(self.d, self.d, generator=g) < 0.5, -1.0, 1.0)
            # Keep the control honest: it must satisfy the same two conditions
            # the algebra does, or it is testing degeneracy rather than
            # arbitrariness. Resample until it does; both are near-certain.
            tries = 0
            while (not non_degenerate(perm, sign)) or has_antipodal_pair(perm, sign):
                perm = torch.stack(
                    [torch.randperm(self.d, generator=g) for _ in range(self.d)]
                )
                sign = torch.where(
                    torch.rand(self.d, self.d, generator=g) < 0.5, -1.0, 1.0
                )
                tries += 1
                if tries > 1000:
                    raise RuntimeError("could not draw a non-degenerate control")
        else:
            if not non_degenerate(perm, sign):
                raise ValueError(f"algebra {algebra!r} is degenerate")
            if has_antipodal_pair(perm, sign):
                raise ValueError(
                    f"algebra {algebra!r} has an antipodal block pair, which "
                    "collapses under an odd activation"
                )

        out, in_, *tail = self.shape
        if out % self.d or in_ % self.d:
            raise ValueError(
                f"{tuple(self.shape)} does not divide by d={self.d} on both "
                "the output and input axes"
            )
        self.tail: Tuple[int, ...] = tuple(tail)
        self.real = nn.Parameter(torch.empty(out // self.d, in_, *tail))
        nn.init.kaiming_uniform_(self.real, a=math.sqrt(5))
        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("sign", sign, persistent=False)
        # The structure matrices themselves, [d, d, d] as (k, p, q). This is the
        # form the expansion is actually computed in - see forward().
        self.register_buffer(
            "structure", structure_matrices(perm, sign).float(), persistent=False
        )

    def forward(self) -> Tensor:
        """Expand the real tensor into ``[out, in, *tail]``.

        WHY A CONTRACTION AND NOT A GATHER. Written out, block ``k`` is
        ``expanded[k, r, g, p] = sum_q P_k[p, q] * real[r, g, q]``, which is a
        contraction against the tiny ``[d, d, d]`` structure tensor - and it is
        the form the paper itself points at: "the product of two hypercomplex
        numbers can also be expressed as a matrix-vector product, which is
        particularly interesting from a computational perspective" (section 2.1).

        The gather this replaced was correct and materially slower, for a reason
        that has nothing to do with arithmetic: the expansion is 4.4e5 element
        ops against the convolution's 1.8 GFLOP, so its cost is memory traffic
        and, above all, the SCATTER-ADD its backward needs. A contraction's
        backward is another contraction.

        Measured on the -o conv shape ``[544, 272, 3]``, as conv fwd+bwd wall
        time against a plain conv on a leaf weight, interleaved min-of-rounds::

            plain conv     25.3 ms   baseline
            contraction    29.0 ms   +14.7%
            d=2 flip       31.3 ms   +23.6%
            gather         41.2 ms   ~+30%   (the original implementation)

        A ``d = 2`` FAST PATH WAS TRIED AND REJECTED, and the reason is worth
        keeping. Two elements admit only two permutations, so a swap is a flip
        and the contraction can be skipped for a broadcast multiply - which
        looks strictly cheaper and is not. Flip, multiply and concatenate is
        three passes over the output; the contraction is one, and its
        ``[d, d, d]`` operand is small enough to stay in cache at any ``d``
        used here. The special case measured NINE POINTS WORSE than the general
        code it was meant to beat, so there is one path for every ``d``.

        Both earlier readings that favoured the flip came from comparing across
        separate benchmark processes, which is how a 30% ordering artifact got
        mistaken for a result. Interleave the candidates and take the min.
        """
        rows, in_ = self.real.shape[0], self.real.shape[1]
        blocks = self.real.reshape(rows, in_ // self.d, self.d, -1)
        out = torch.einsum("kpq,rgqs->krgps", self.structure, blocks)
        return out.reshape(self.d * rows, in_, *self.tail)

    def extra_repr(self) -> str:
        return f"algebra={self.algebra}, d={self.d}, " + super().extra_repr()


class LowRankExpansion(Expansion):
    """``W = U V``, at the parameter budget an ``AlgebraExpansion`` would spend.

    ``rank`` is solved for, never set: ``r = round(out * in * tail / d / (out +
    in * tail))``, floored at 1. So this arm is matched to the algebra arm by
    construction at whatever shape it lands on.
    """

    def __init__(self, shape: Sequence[int], d: int = 2, tag: str = "") -> None:
        super().__init__(shape, tag)
        out, in_, *tail = self.shape
        fan = in_ * math.prod(tail) if tail else in_
        budget = out * fan // d
        self.rank = max(1, round(budget / (out + fan)))
        self.d = int(d)
        self.tail: Tuple[int, ...] = tuple(tail)
        self.u = nn.Parameter(torch.empty(out, self.rank))
        self.v = nn.Parameter(torch.empty(self.rank, fan))
        # INIT IS SOLVED, NOT ADJUSTED. The PRODUCT has to land on the variance
        # the module being replaced would have had, which for torch's default
        # `kaiming_uniform_(a=sqrt(5))` on a Linear/Conv weight is
        # ``gain^2 / fan`` with ``gain^2 = 2 / (1 + 5) = 1/3``. Since
        # ``Var(UV) = rank * Var(U) * Var(V)``, splitting evenly gives
        #
        #     s = (gain^2 / (fan * rank)) ** 0.25
        #
        # and ``rank * s^4 == gain^2 / fan`` by construction, at any rank.
        #
        # The previous version init'd each factor with `kaiming_uniform_` and
        # then multiplied both by `rank^-0.25`. `kaiming_uniform_` already
        # normalizes by each factor's OWN fan-in, so those multipliers were a
        # second, unearned `rank^-0.5` on the product: measured 0.045x the
        # correct scale at rank 163, i.e. six conv layers starting 22x too
        # quiet. That is the same failure `praxis/dense/peer.py:init_weights`
        # documents for Xavier on a lookup bank - an init scale falling with a
        # dimension that is not a fan - and it invalidated the first -r run.
        # `test_init_scale_matches_the_replaced_module` now pins this.
        std = (self.GAIN_SQ / (fan * self.rank)) ** 0.25
        bound = math.sqrt(3.0) * std  # uniform with this std
        with torch.no_grad():
            self.u.uniform_(-bound, bound)
            self.v.uniform_(-bound, bound)

    def forward(self) -> Tensor:
        return (self.u @ self.v).view(*self.shape)

    def extra_repr(self) -> str:
        return f"rank={self.rank}, " + super().extra_repr()


def _algebra(name: str, randomize: bool = False):
    def build(shape: Sequence[int], tag: str = "") -> Expansion:
        return AlgebraExpansion(shape, algebra=name, tag=tag, randomize=randomize)

    return build


def _lowrank(d: int):
    def build(shape: Sequence[int], tag: str = "") -> Expansion:
        return LowRankExpansion(shape, d=d, tag=tag)

    return build


# The arms of the experiment, as registry entries. Adding an arm is a line here,
# never a new integration site.
EXPANSION_REGISTRY = {
    # d = 2. Complex numbers: non-degenerate, named by the paper as covered by
    # its approximation guarantee, and the only halving that lands exactly on a
    # GLU's two-way output split.
    "complex": _algebra("complex"),
    # d = 4. The paper's own algebra. A 75% cut, not a halving.
    "quaternion": _algebra("quaternion"),
    # Controls, same shape and budget as their algebra twin.
    "random": _algebra("complex", randomize=True),
    "random4": _algebra("quaternion", randomize=True),
    "lowrank": _lowrank(2),
    "lowrank4": _lowrank(4),
}
