"""Per-objective gradients at the trunk output, and the anchor they compare to.

Shared by :mod:`praxis.losses.conflict` (which measures) and
:mod:`praxis.strategies` (which acts). Both need the same two things: one row
per loss term at the activation the head classifies, and agreement on which row
is the task.

WHY THE TRUNK OUTPUT AND NOT THE PARAMETERS. The honest Jacobian is w.r.t.
shared parameters and needs a full backward per objective. Differentiating
w.r.t. the trunk's output ACTIVATION needs only a backward through the head and
answers the same question: by the chain rule every shared parameter's gradient
factors through that tensor. What it cannot see is conflict arising INSIDE the
trunk, or between terms that act upstream of it - an encoder's VQ commitment
loss has no path to this tensor and gets no row.

WHY THE ANCHOR IS NOT ALWAYS "main". Under a surgical head (prismatic9) the
mixture cross-entropy trains only the gate: every arm is detached in the blend
and the gate's input is detached too, so ``main`` has NO path to the trunk at
all. Measured on prismatic7/8 the mixture reaches the trunk; on prismatic9
``autograd.grad(main, hidden_states)`` returns None. The row that carries the
task into the trunk there is ``arm_surgery``, the arms' PCGrad-combined
gradient. Anchoring on a term with no gradient is why every ``conflict_*``
series stayed dark through abstractinator-u.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

# Preference order for the anchor: the task objective, then the surgical head's
# single row into the trunk when the mixture CE has been detached from it by
# design. First one with a live gradient wins.
ANCHOR_PREFERENCE: Tuple[str, ...] = ("main", "arm_surgery")

# Below this a gradient is numerically zero and any ratio or cosine against it
# is noise.
MIN_NORM: float = 1e-12


def trunk_gradients(
    loss_dict: Dict[str, Any], wrt: Tensor
) -> Dict[str, Tensor]:
    """``{name: dL/dwrt}`` for every term that actually pulls on ``wrt``.

    A term with no path to the shared representation is ABSENT rather than
    zero: it does not compete for the trunk, and saying so by omission is the
    result. Terms outside the retained graph are skipped the same way -
    diagnostics and folds alike must never take a run down.
    """
    out: Dict[str, Tensor] = {}
    for name, term in loss_dict.items():
        if not isinstance(term, Tensor) or not term.requires_grad:
            continue
        try:
            g = torch.autograd.grad(term, wrt, retain_graph=True, allow_unused=True)[0]
        except RuntimeError:
            continue
        if g is None:
            continue
        g = g.detach().float()
        if float(g.norm()) < MIN_NORM:
            continue
        out[name] = g
    return out


def resolve_anchor(names) -> Optional[str]:
    """Which of the live rows is the task. None when none of them is."""
    live = set(names)
    for candidate in ANCHOR_PREFERENCE:
        if candidate in live:
            return candidate
    return None


def usable(wrt: Any) -> bool:
    """Whether a row-wise measurement can run against ``wrt`` at all.

    Dynamo cannot trace ``autograd.grad``, so under compile this reports False
    and every caller falls back to its uncorrected path - the same convention
    the compute profiler takes for the same reason.
    """
    if torch.compiler.is_compiling():
        return False
    if not torch.is_grad_enabled():
        return False
    return isinstance(wrt, Tensor) and wrt.requires_grad
