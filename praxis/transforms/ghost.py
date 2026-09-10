r"""Ghost features: extra weight blocks derived from weights you already have.

THE NAME IS THEIRS. Vieira Neto & Valle, "Ghost Features and Spooky Transfer
Learning for Hypercomplex-Valued Neural Networks" (arXiv:2608.07735, 2026),
following GhostNet's ghost feature maps (Han et al., CVPR 2020). This module
implements their construction, so it keeps their word for it. The surrounding
namespace is ``praxis.transforms`` because a tree-walking parameter rewrite is a
general shape and ghosting is one instance of it.

Not to be confused with ghostmax (``praxis/attention/causal.py``), a phantom key
prepended to the softmax denominator. Nothing connects the two but the word.

APPLIED AS AN IN-PLACE PARAMETRIZATION, not a wrapper, so it works on any module
owning a ``weight`` Parameter and COMPOSES with the parameter-merging routers.
Swapping a module for a wrapper changes the object at that qualname, and
anything holding a reference to the original keeps the original - SMEAR
registers its ``MergedLinear`` in ``self.wrappers`` AND at the block qualname,
so replacing the qualname would leave the router driving a module the block no
longer uses. ``register_parametrization`` mutates in place: same object, same
identity, ``isinstance`` unchanged. So ghost and SMEAR stack:

    y = expand(real) @ x  +  sum_e c_be B_e (A_e x)
        \_______________/     \______________________/
         ghost-derived base     SMEAR's learned deviations, untouched

WHY ``right_inverse`` MATTERS. PyTorch calls it once at registration to turn the
module's EXISTING weight into the stored tensor. Our ``P_k`` are involutions, so
``mean_k P_k(W_k)`` is the least-squares real tensor for an incoming ``W``: the
ghosted module starts at the closest representable point to whatever init its
host already chose. Every module keeps its own init convention for free and no
scale factor is picked here.

The stored tensor is ``parametrizations.weight.original`` at
``[out // d, in, *tail]`` - genuinely smaller, and the only parameter for that
weight.
"""

from __future__ import annotations

import math
import zlib
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch import Tensor

from praxis.transforms.algebra import (
    ALGEBRAS,
    has_antipodal_pair,
    non_degenerate,
    structure_matrices,
    tables,
)

# Preference order for `auto`. Larger d is a deeper cut, so try the deepest that
# divides the tensor on both axes. The cyclic entries exist so d is not limited
# to the powers of two the named algebras cover: PEER's banks are [729, 272] and
# 729 = 3^6, which only d = 3 divides.
AUTO_ORDER: Tuple[str, ...] = ("quaternion", "cyclic3", "complex")


def _stable_seed(tag: str) -> int:
    """Reproducible across processes, unlike ``hash()``, which is salted."""
    return zlib.crc32(tag.encode("utf-8")) & 0x7FFFFFFF


def _random_tables(d: int, tag: str, algebra: str) -> Tuple[Tensor, Tensor]:
    """An arbitrary FROZEN signed permutation, seeded by the target's qualname.

    The control for "does the ALGEBRA matter, given that a fixed full-rank
    input-axis expansion helps at all". It must satisfy the same two conditions
    the named algebras do, or it tests degeneracy rather than arbitrariness, so
    it resamples until it does.
    """
    g = torch.Generator().manual_seed(_stable_seed(f"{tag}|{algebra}"))
    for _ in range(1000):
        perm = torch.stack([torch.randperm(d, generator=g) for _ in range(d)])
        sign = torch.where(torch.rand(d, d, generator=g) < 0.5, -1.0, 1.0)
        if non_degenerate(perm, sign) and not has_antipodal_pair(perm, sign):
            return perm, sign
    raise RuntimeError("could not draw a non-degenerate control")


class GhostExpansion(nn.Module):
    """``original`` is the real tensor; ``weight`` is its signed-permutation expansion."""

    def __init__(
        self, algebra: str, shape: Sequence[int], tag: str = "", randomize: bool = False
    ) -> None:
        super().__init__()
        perm, sign = tables(algebra)
        if randomize:
            perm, sign = _random_tables(int(perm.size(0)), tag, algebra)
        self.algebra = f"random_d{int(perm.size(0))}" if randomize else algebra
        self.d = int(perm.size(0))
        if not non_degenerate(perm, sign):
            raise ValueError(f"algebra {self.algebra!r} is degenerate")
        if has_antipodal_pair(perm, sign):
            raise ValueError(f"algebra {self.algebra!r} has an antipodal block pair")
        out, in_, *tail = tuple(int(s) for s in shape)
        if out % self.d or in_ % self.d:
            raise ValueError(
                f"{tuple(shape)} does not divide by d={self.d} on both axes"
            )
        self.full_shape: Tuple[int, ...] = (out, in_, *tail)
        self.register_buffer(
            "structure", structure_matrices(perm, sign).float(), persistent=False
        )

    @property
    def real_numel(self) -> int:
        out, in_, *tail = self.full_shape
        return (out // self.d) * in_ * math.prod(tail)

    def forward(self, original: Tensor) -> Tensor:
        rows, in_ = original.shape[0], original.shape[1]
        tail = tuple(original.shape[2:])
        blocks = original.reshape(rows, in_ // self.d, self.d, -1)
        out = torch.einsum("kpq,rgqs->krgps", self.structure, blocks)
        return out.reshape(self.d * rows, in_, *tail)

    def right_inverse(self, weight: Tensor) -> Tensor:
        """Least-squares real tensor for an incoming full weight.

        Each ``P_k`` used here is an involution, so ``P_k^{-1} = P_k`` and the
        least-squares fit is the mean of the back-transformed blocks. This is the
        same estimator the free gate uses to ask how close a TRAINED weight sits
        to the ghost manifold (see next/ghost_features.md).
        """
        out, in_ = weight.shape[0], weight.shape[1]
        tail = tuple(weight.shape[2:])
        rows = out // self.d
        blocks = weight.reshape(out, in_ // self.d, self.d, -1)
        est = (
            sum(
                torch.einsum(
                    "pq,rgqs->rgps",
                    self.structure[k],
                    blocks[k * rows : (k + 1) * rows],
                )
                for k in range(self.d)
            )
            / self.d
        )
        return est.reshape(rows, in_, *tail)

    def extra_repr(self) -> str:
        # PyTorch convention: key=value only, one line. No prose, no arrows.
        return f"algebra={self.algebra!r}, d={self.d}, shape={self.full_shape}"


def pick_algebra(shape: Sequence[int], requested: str) -> Optional[str]:
    """Resolve ``requested`` against a shape; ``None`` when nothing fits.

    ``auto`` walks AUTO_ORDER and takes the first algebra whose ``d`` divides both
    the output and input axes. Derived from the shape, never swept, so it stays
    inside the no-tuning rule - but it does mean a broad `auto` profile mixes
    algebras across sites, which costs attribution. Use a named algebra when the
    question is which algebra.
    """
    out, in_, *_ = tuple(int(s) for s in shape)
    names = AUTO_ORDER if requested == "auto" else (requested,)
    for name in names:
        d = len(ALGEBRAS[name][0])
        if out % d == 0 and in_ % d == 0:
            return name
    return None


def ghost_parameter(
    module: nn.Module, name: str, algebra: str, tag: str = "", randomize: bool = False
) -> GhostExpansion:
    """Parametrize ``module.<name>`` as a ghost expansion, in place.

    ``unsafe=True`` is required and is not a shortcut: it tells PyTorch the
    parametrization CHANGES the tensor's shape, which is the entire point here.
    """
    weight = getattr(module, name)
    param = GhostExpansion(algebra, tuple(weight.shape), tag=tag, randomize=randomize)
    parametrize.register_parametrization(module, name, param, unsafe=True)
    _compact_repr(module, name)
    return param


def _compact_repr(module: nn.Module, name: str) -> None:
    """Keep a ghosted leaf module on ONE line, in PyTorch's own key=value style.

    ``register_parametrization`` renders as a seven-line nest -
    ``ParametrizedLinear`` wrapping a ``ModuleDict`` wrapping a
    ``ParametrizationList`` - which is structurally accurate and useless to read
    when every Linear in the model carries one. The model repr is a tab in the web
    UI and gets scanned constantly, so a ghosted Linear renders as::

        Linear(in_features=272, out_features=90, bias=True, ghost=complex(d=2))

    Original class name, original arguments, one more key=value pair. No prose, no
    arrows. PyTorch mints a fresh subclass per parametrized module, so patching it
    is local to this instance's type.
    """
    cls = type(module)
    if getattr(cls, "_ghost_repr_patched", False):
        return
    base_extra = cls.extra_repr

    def extra_repr(self) -> str:
        parts = [base_extra(self)]
        for pname, plist in self.parametrizations.items():
            for entry in plist:
                if isinstance(entry, GhostExpansion):
                    tag = f"{entry.algebra}(d={entry.d})"
                    parts.append(
                        f"ghost={tag}" if pname == "weight" else f"ghost_{pname}={tag}"
                    )
        return ", ".join(x for x in parts if x)

    def __repr__(self) -> str:
        # The only children are the parametrization machinery, and its content is
        # already in extra_repr, so there is nothing left to nest.
        return f"{type(self).__bases__[0].__name__}({extra_repr(self)})"

    cls.extra_repr = extra_repr
    cls.__repr__ = __repr__
    cls._ghost_repr_patched = True
