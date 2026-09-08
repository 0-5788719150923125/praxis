"""Ghost features: extra weight blocks derived from weights you already have.

A pure function over an assembled model, in the shape ``praxis/routers/smear.py``
established for its own target discovery: walk the module tree, apply a named
profile of include/exclude rules, and rebind what matches. Nothing here is
site-specific, which is the point - moving the experiment to a different tensor
is a regex in ``GHOST_REGISTRY``, not a new implementation. The alternative,
patching each host module in place, would have meant a separate change in the
encoder, the FFN, the MTP bank and the head to ask the same question four times.

The mechanism is structured weight tying, not quaternion arithmetic. See
``praxis/ghost/algebra.py`` for the derivation and for the one guarantee the
source paper (arXiv:2608.07735) actually offers.

WHY THIS DOES NOT REUSE ``discover_targets``. That walker carries two structural
exclusions earned by parameter-MERGING routers: ``MERGE_OPAQUE`` subtrees, and
tied-by-reference parameters that two coefficient rows would fight over. Neither
argument transfers. ``MERGE_OPAQUE`` says "this module already routes itself per
token, so a per-batch merge around it buys nothing" - a statement about routing
granularity that has nothing to say about weight tying. Reusing the flag would
make PEER un-ghostable for a reason that does not apply to ghosting. The shared
piece is ``TargetSpec``, which is just the regex rules, and that IS reused.

Tied-by-reference IS still handled, by id, because ghosting one member of a tied
pair and not the other silently unties them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch.nn as nn

from praxis.ghost.expansions import EXPANSION_REGISTRY
from praxis.ghost.modules import WRAPPERS
from praxis.routers.targeting import TargetSpec

__all__ = ["GHOST_REGISTRY", "GhostProfile", "GhostStats", "ghostify"]


@dataclass(frozen=True)
class GhostProfile:
    """Where to ghost, and with which expansion rule."""

    spec: TargetSpec
    rule: str


# The encoder ConvBlock stack. 39.1% of abstractinator-o's parameters live in
# these six Conv1d weights, more than three times the FFN's share, and the
# block already ends in a GLU multiply plus a `proj` Linear - so the mixer the
# construction needs is bought and paid for. Both stacks are targeted: the
# abstractinator's compress and reconstruct halves are symmetric, and taking
# only one would halve the signal for no gain in attribution.
_CONV = TargetSpec(include=(r"encoder\.(encoder|decoder)\.layers\.\d+\.conv",))

GHOST_REGISTRY: Dict[str, GhostProfile] = {
    "conv_complex": GhostProfile(_CONV, "complex"),
    "conv_quaternion": GhostProfile(_CONV, "quaternion"),
    "conv_random": GhostProfile(_CONV, "random"),
    "conv_lowrank": GhostProfile(_CONV, "lowrank"),
}


@dataclass
class GhostStats:
    """What a ghostify pass actually did, for the build log and for tests."""

    profile: str
    rule: str
    targets: List[Tuple[str, int, int]]  # qualname, params before, params after
    skipped: Dict[str, int]

    @property
    def before(self) -> int:
        return sum(b for _, b, _ in self.targets)

    @property
    def after(self) -> int:
        return sum(a for _, _, a in self.targets)

    def describe(self, model_total: int = 0) -> str:
        """``model_total`` is the POST-ghost count, which is what the caller
        has in hand after the pass; the pre-ghost figure is recovered from the
        saving rather than requiring the caller to measure twice."""
        lines = [
            f"[GHOST] {self.profile} ({self.rule}): {len(self.targets)} targets, "
            f"{self.before:,} -> {self.after:,} parameters "
            f"({self.before - self.after:,} saved)"
        ]
        for name, before, after in self.targets:
            lines.append(f"  {name:<44} {before:>10,} -> {after:>9,}")
        if model_total:
            saved = self.before - self.after
            baseline = model_total + saved
            lines.append(
                f"  model {baseline:,} -> {model_total:,} "
                f"({100 * saved / baseline:.1f}% cut)"
            )
        drops = ", ".join(f"{k}={v}" for k, v in self.skipped.items() if v)
        if drops:
            lines.append(f"  skipped: {drops}")
        return "\n".join(lines)


def ghostify(model: nn.Module, profile: str) -> GhostStats:
    """Replace every module the profile matches with its ghost equivalent.

    Mutates ``model`` in place and returns what it did. ``profile`` of ``none``
    (or empty) is a no-op returning empty stats, so the call site needs no
    conditional.
    """
    if not profile or profile == "none":
        return GhostStats("none", "none", [], {})
    if profile not in GHOST_REGISTRY:
        raise ValueError(
            f"Unknown ghost profile {profile!r}; known: {sorted(GHOST_REGISTRY)}"
        )

    entry = GHOST_REGISTRY[profile]
    factory = EXPANSION_REGISTRY[entry.rule]

    targets: List[Tuple[str, int, int]] = []
    skipped: Dict[str, int] = {"unmatched": 0, "unwrappable": 0, "shared": 0, "shape": 0}
    seen_ids: set = set()

    # Materialize the walk before mutating: replacing a submodule during
    # named_modules() would hand us our own wrappers.
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if type(module) in WRAPPERS
    ]

    for name, module in candidates:
        if not entry.spec.matches(name):
            skipped["unmatched"] += 1
            continue
        weight = getattr(module, "weight", None)
        if weight is None or not isinstance(weight, nn.Parameter):
            skipped["unwrappable"] += 1
            continue
        if id(weight) in seen_ids:
            # Tied by reference. Ghosting one side and not the other unties
            # them silently, which is a different experiment than the one asked
            # for, so neither side is touched.
            skipped["shared"] += 1
            continue

        before = weight.numel()
        try:
            wrapper = WRAPPERS[type(module)](module, factory, tag=name)
        except ValueError:
            # The shape does not divide by d, or the module is a form the
            # wrapper does not reproduce. Reported, never silently skipped.
            skipped["shape"] += 1
            continue

        seen_ids.add(id(weight))
        model.set_submodule(name, wrapper)
        targets.append((name, before, wrapper.expansion.real_numel))

    if not targets:
        raise ValueError(
            f"Ghost profile {profile!r} matched nothing. Skips: {skipped}"
        )
    return GhostStats(profile, entry.rule, targets, skipped)
