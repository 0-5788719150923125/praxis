"""Ghost features: extra weight blocks derived from weights you already have.

A pure function over an assembled model, in the shape ``praxis/routers/smear.py``
established for target discovery: walk the module tree, apply a named profile of
include/exclude rules, and transform what matches. Nothing here is site-specific,
which is the point - moving the experiment to a different tensor is a regex in
``TRANSFORM_REGISTRY``, not a new implementation.

The mechanism is structured weight tying, not quaternion arithmetic. See
``praxis/ghost/algebra.py`` for the derivation and for the one guarantee the
source paper (arXiv:2608.07735) actually offers, and
``praxis/ghost/parametrization.py`` for why the transform is applied IN PLACE.

IT COMPOSES, IT DOES NOT EXCLUDE. Ghosting is a parametrization of a weight, and
the parameter-merging routers are a deviation ON a weight, so they stack::

    y = expand(real) @ x  +  sum_e c_be B_e (A_e x)
        \\_______________/     \\______________________/
         ghost-derived base     SMEAR's learned deviations, untouched

SMEAR's ``MergedLinear`` is therefore a target like any other, and the fact that
SMEAR holds its own reference to that module is exactly why the transform must
mutate rather than wrap.

WHY THIS DOES NOT REUSE ``discover_targets``. That walker excludes
``MERGE_OPAQUE`` subtrees, which is a statement about ROUTING GRANULARITY ("this
already routes per token, so a per-batch merge around it buys nothing") with
nothing to say about weight tying. PEER sets that flag, and PEER's banks are the
largest tensor group in the decoder - reusing the flag would make them
permanently unreachable for a reason that does not apply. ``GHOST_OPAQUE`` is the
separate flag, and it means something narrower: this module's parameters are
addressed by a name captured at construction, or rewritten in place by an inner
loop, so a derived tensor has nowhere to be written back to.

Tied-by-reference parameters ARE still skipped, by id: ghosting one member of a
tied pair and not the other silently unties them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.parameter import UninitializedParameter

from praxis.transforms.ghost import ghost_parameter, pick_algebra
from praxis.transforms.targeting import TargetSpec

__all__ = [
    "TRANSFORM_REGISTRY",
    "MIN_TARGET_NUMEL",
    "TransformProfile",
    "TransformStats",
    "apply_transform",
]


@dataclass(frozen=True)
class TransformProfile:
    """Where to ghost, with which algebra, and what to spare."""

    spec: TargetSpec
    # An algebra name from praxis.transforms.algebra, or "auto" to take the deepest
    # cut whose d divides the tensor on both axes.
    algebra: str
    # Frozen arbitrary signed permutation instead of the algebra's. The control
    # for "does the ALGEBRA matter", not an ablation.
    randomize: bool = False
    # Spare tensors with a `vocab_size` dimension. False only where tying the
    # readout IS the experiment.
    skip_vocab: bool = True


# The encoder ConvBlock stack: `RMSNorm -> Conv1d(d, 2d) -> GLU -> proj`, so the
# mixer the construction needs is already bought. 39% of the model.
_CONV = TargetSpec(include=(r"encoder\.(encoder|decoder)\.layers\.\d+\.conv",))
# The MTP bank's per-depth projections - the tightest large tensors in the model
# by the spectral survey (sv99/out 0.83-0.89 against the convs' 0.29-0.33).
_MTP = TargetSpec(include=(r"mtp\.bank\.depths\.\d+\.projection",))
# Everything eligible.
_ALL = TargetSpec()

# Below this, ghosting is overhead with no saving: the expansion costs a kernel
# per forward and per backward whatever the size. Matches
# targeting.DENSE_DELTA_MAX_NUMEL, which draws the same line for the same reason.
MIN_TARGET_NUMEL: int = 4096

TRANSFORM_REGISTRY: Dict[str, TransformProfile] = {
    "ghost_conv_complex": TransformProfile(_CONV, "complex"),
    "ghost_conv_quaternion": TransformProfile(_CONV, "quaternion"),
    "ghost_conv_random": TransformProfile(_CONV, "complex", randomize=True),
    "ghost_mtp_complex": TransformProfile(_MTP, "complex"),
    "ghost_mtp_random": TransformProfile(_MTP, "complex", randomize=True),
    "ghost_all_complex": TransformProfile(_ALL, "complex"),
    "ghost_all_quaternion": TransformProfile(_ALL, "quaternion"),
    "ghost_all_random": TransformProfile(_ALL, "complex", randomize=True),
    # Deepest cut each tensor admits. Maximizes coverage; costs attribution,
    # because a difference could be the algebra or the depth.
    "ghost_all_auto": TransformProfile(_ALL, "auto"),
    # Spares nothing, readout included. Expect the LM head to be what breaks.
    "ghost_all_greedy_complex": TransformProfile(_ALL, "complex", skip_vocab=False),
}


@dataclass
class TransformStats:
    """What a apply_transform pass actually did, for the build log and for tests."""

    profile: str
    algebra: str
    targets: List[Tuple[str, int, int, str]]  # qualname, before, after, algebra
    skipped: Dict[str, int]
    # reason -> the qualnames dropped for it. A broad profile that silently
    # covers a third of what you think it covers is worse than one that covers
    # nothing, so the misses are kept, not just counted.
    missed: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def before(self) -> int:
        return sum(b for _, b, _, _ in self.targets)

    @property
    def after(self) -> int:
        return sum(a for _, _, a, _ in self.targets)

    def describe(self, model_before: int = 0, model_after: int = 0) -> str:
        """Both model totals are MEASURED by the caller, never inferred here.

        An earlier version took only the post-ghost count and reconstructed the
        baseline as ``after + saved``. That is wrong whenever a transform leaves a
        tensor in place that it did not tie, so the two disagreed and the log
        quietly misreported the baseline. The [GHOST] block is what a reader
        checks to confirm the run is the experiment.
        """
        lines = [
            f"[GHOST] {self.profile} ({self.algebra}): {len(self.targets)} targets, "
            f"{self.before:,} -> {self.after:,} parameters "
            f"({self.before - self.after:,} saved)"
        ]
        for name, before, after, alg in self.targets:
            lines.append(f"  {name:<46} {before:>9,} -> {after:>9,}  {alg}")
        if model_before and model_after:
            cut = model_before - model_after
            lines.append(
                f"  model {model_before:,} -> {model_after:,} "
                f"({100 * cut / model_before:.1f}% cut)"
            )
            residual = (self.before - self.after) - cut
            if residual:
                lines.append(f"  UNRECONCILED: {residual:,} (report this)")
        drops = ", ".join(f"{k}={v}" for k, v in sorted(self.skipped.items()) if v)
        if drops:
            lines.append(f"  skipped: {drops}")
        return "\n".join(lines)


def apply_transform(model: nn.Module, profile: str) -> TransformStats:
    """Parametrize every eligible weight the profile matches, in place.

    Mutates ``model`` and returns what it did. ``profile`` of ``none`` (or empty)
    is a no-op, so the call site needs no conditional.
    """
    if not profile or profile == "none":
        return TransformStats("none", "none", [], {}, {})
    if profile not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown ghost profile {profile!r}; known: {sorted(TRANSFORM_REGISTRY)}"
        )

    entry = TRANSFORM_REGISTRY[profile]
    skipped: Dict[str, int] = {}
    missed: Dict[str, List[str]] = {}

    def drop(reason: str, name: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1
        missed.setdefault(reason, []).append(name)

    targets: List[Tuple[str, int, int, str]] = []
    seen_ids: set = set()
    vocab = getattr(getattr(model, "config", None), "vocab_size", None)

    opaque = [
        name
        for name, module in model.named_modules()
        if getattr(module, "GHOST_OPAQUE", False)
    ]

    def under_opaque(name: str) -> bool:
        return any(name == o or name.startswith(o + ".") for o in opaque)

    # Materialize the walk before mutating. Parametrization inserts submodules,
    # so iterating live would visit our own machinery.
    # `"weight" in module._parameters`, not `isinstance(module.weight, Parameter)`.
    # Several modules here expose `weight` as a derived PROPERTY forwarding to a
    # child (AdditiveEmbedding does), and an attribute check accepts those - then
    # register_parametrization rejects them, because there is no registered
    # parameter of that name to parametrize. Ask the registry, not the attribute.
    # Already-parametrized modules are included DELIBERATELY. Parametrizing moves
    # `weight` out of `_parameters` (to `parametrizations.weight.original`), so a
    # second pass would otherwise not see them at all and would report nothing -
    # the same silent-coverage gap as the type filter this replaced. They are
    # counted and named as `already_parametrized`, then left alone.
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module._parameters.get("weight"), nn.Parameter)
        or parametrize.is_parametrized(module, "weight")
    ]

    for name, module in candidates:
        if parametrize.is_parametrized(module, "weight"):
            drop("already_parametrized", name)
            continue
        if under_opaque(name):
            drop("opaque", name)
            continue
        if not entry.spec.matches(name):
            drop("unmatched", name)
            continue
        weight = module.weight
        if isinstance(weight, UninitializedParameter):
            drop("lazy", name)
            continue
        if not weight.requires_grad:
            drop("frozen", name)
            continue
        if id(weight) in seen_ids:
            drop("shared", name)
            continue
        if weight.numel() < MIN_TARGET_NUMEL:
            drop("too_small", name)
            continue
        if weight.dim() < 2:
            drop("rank1", name)
            continue
        if entry.skip_vocab and vocab is not None and vocab in tuple(weight.shape):
            drop("vocab", name)
            continue
        algebra = pick_algebra(tuple(weight.shape), entry.algebra)
        if algebra is None:
            # No d divides the tensor on BOTH axes. PEER's default banks are the
            # canonical case: [729, 272] with gcd(729, 272) = 1.
            drop("indivisible", name)
            continue

        before = weight.numel()
        seen_ids.add(id(weight))
        param = ghost_parameter(
            module, "weight", algebra, tag=name, randomize=entry.randomize
        )
        targets.append((name, before, param.real_numel, param.algebra))

    if not targets:
        raise ValueError(f"Ghost profile {profile!r} matched nothing. Skips: {skipped}")
    return TransformStats(profile, entry.algebra, targets, skipped, missed)
