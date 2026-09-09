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

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch.nn as nn

from praxis.ghost.expansions import EXPANSION_REGISTRY
from praxis.ghost.modules import WRAPPERS
from torch.nn.parameter import UninitializedParameter

from praxis.routers.targeting import TargetSpec

__all__ = ["GHOST_REGISTRY", "GhostProfile", "GhostStats", "ghostify"]


@dataclass(frozen=True)
class GhostProfile:
    """Where to ghost, and with which expansion rule."""

    spec: TargetSpec
    rule: str
    # Spare tensors with a `vocab_size` dimension in their shape. False only for
    # the greedy profiles, where tying the readout IS the experiment.
    skip_vocab: bool = False


# The encoder ConvBlock stack. 39.1% of abstractinator-o's parameters live in
# these six Conv1d weights, more than three times the FFN's share, and the
# block already ends in a GLU multiply plus a `proj` Linear - so the mixer the
# construction needs is bought and paid for. Both stacks are targeted: the
# abstractinator's compress and reconstruct halves are symmetric, and taking
# only one would halve the signal for no gain in attribution.
_CONV = TargetSpec(include=(r"encoder\.(encoder|decoder)\.layers\.\d+\.conv",))

# The MTP bank's five per-depth projections. Chosen by MEASUREMENT, not by size:
# the spectral survey of -o's checkpoint puts these at sv99/out 0.83-0.89, the
# tightest large tensors in the model, against 0.29-0.33 for five of the six
# ConvBlock stacks. That matters because -q and -t showed the conv site cannot
# discriminate: a 50% cut and a 75% cut both tie the baseline, so any structure
# ties there and the comparison has no power. A site that uses its rank can tell
# a good constraint from a bad one.
_MTP = TargetSpec(include=(r"mtp\.bank\.depths\.\d+\.projection",))

# Everything eligible. The point of a generic transform is that "try it
# everywhere" is a profile rather than a project, so these exist to make the
# mechanism available at scale for future work - not because blanket application
# is expected to pay. The original design note rejected blanket application for a
# good reason (the mixer that makes ghost channels usable is not free), and
# nothing since has overturned that.
_ALL = TargetSpec()

# Below this, ghosting is not worth its own overhead: the expansion is a kernel
# per forward and per backward regardless of size. Matches
# targeting.DENSE_DELTA_MAX_NUMEL, which draws the same small/large line for the
# same reason.
MIN_TARGET_NUMEL: int = 4096

GHOST_REGISTRY: Dict[str, GhostProfile] = {
    "mtp_complex": GhostProfile(_MTP, "complex"),
    "mtp_random": GhostProfile(_MTP, "random"),
    # Broad profiles. `all_*` spares tensors carrying a vocab dimension - the
    # embeddings, the LM head, and anything tied to either. That is not
    # squeamishness: it is the same shape-based rule
    # `praxis/optimization/__init__.py:_split_muon_params` already uses to keep
    # Muon off those matrices, and for the same reason - they are lookup and
    # readout tables rather than interior geometry, and tying token k to token
    # k + V/2 by a fixed signed permutation is a different (and much stronger)
    # claim than tying two halves of a hidden projection.
    "all_complex": GhostProfile(_ALL, "complex", skip_vocab=True),
    "all_quaternion": GhostProfile(_ALL, "quaternion", skip_vocab=True),
    "all_random": GhostProfile(_ALL, "random", skip_vocab=True),
    # ...and the version that spares nothing, for when that claim is the
    # experiment. Expect the LM head to be the thing that breaks.
    "all_greedy_complex": GhostProfile(_ALL, "complex", skip_vocab=False),
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
    # reason -> the qualnames dropped for it. A broad profile that silently
    # covers a third of what you think it covers is worse than one that covers
    # nothing, so the misses are kept, not just counted.
    missed: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def before(self) -> int:
        return sum(b for _, b, _ in self.targets)

    @property
    def after(self) -> int:
        return sum(a for _, _, a in self.targets)

    def describe(self, model_before: int = 0, model_after: int = 0) -> str:
        """Both model totals are MEASURED by the caller, never inferred here.

        An earlier version took only the post-ghost count and reconstructed the
        baseline as ``after + saved``. That is wrong whenever a wrapper carries a
        tensor it does not tie - biases are kept real by design - so the two
        disagreed and the log line quietly misreported the baseline. The [GHOST]
        block is what a reader checks to confirm the run is the experiment, so it
        measures both ends rather than deriving one from the other."""
        lines = [
            f"[GHOST] {self.profile} ({self.rule}): {len(self.targets)} targets, "
            f"{self.before:,} -> {self.after:,} parameters "
            f"({self.before - self.after:,} saved)"
        ]
        for name, before, after in self.targets:
            lines.append(f"  {name:<44} {before:>10,} -> {after:>9,}")
        if model_before and model_after:
            cut = model_before - model_after
            lines.append(
                f"  model {model_before:,} -> {model_after:,} "
                f"({100 * cut / model_before:.1f}% cut)"
            )
            # Tensors the wrappers carried over rather than tied - biases, and
            # anything else a future wrapper decides to keep real.
            carried = (self.before - self.after) - cut
            if carried:
                lines.append(f"  carried over untied: {carried:,}")
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
        return GhostStats("none", "none", [], {}, {})
    if profile not in GHOST_REGISTRY:
        raise ValueError(
            f"Unknown ghost profile {profile!r}; known: {sorted(GHOST_REGISTRY)}"
        )

    entry = GHOST_REGISTRY[profile]
    factory = EXPANSION_REGISTRY[entry.rule]

    targets: List[Tuple[str, int, int]] = []
    skipped: Dict[str, int] = {}
    missed: Dict[str, List[str]] = {}

    def drop(reason: str, name: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1
        missed.setdefault(reason, []).append(name)

    seen_ids: set = set()
    vocab = getattr(getattr(model, "config", None), "vocab_size", None)

    # Subtrees that opt out, by asking the registered class rather than matching
    # a hardcoded name list - the idiom targeting.py uses for MERGE_OPAQUE, and
    # `_wants_expert_bank` in praxis/decoders/base.py before it. A module must
    # declare GHOST_OPAQUE when its parameters are addressed by a name captured
    # at construction, or rewritten in place by an inner loop: both survive the
    # shape check and then fail, because ghosting RENAMES (`w` -> `expansion.
    # real`) and DERIVES (nothing to write back to). NeuralMemory is the case
    # this was found on - its fast-weight Adam looks its own tensors up through
    # `self._param_names`.
    opaque = [
        name
        for name, module in model.named_modules()
        if getattr(module, "GHOST_OPAQUE", False)
    ]

    def under_opaque(name: str) -> bool:
        return any(name == o or name.startswith(o + ".") for o in opaque)

    # Materialize the walk before mutating: replacing a submodule during
    # named_modules() would hand us our own wrappers. Exact type match, never
    # isinstance - a subclass may have a forward the wrapper does not reproduce,
    # and silently running the parent's forward is the worst failure available.
    candidates = []
    for name, module in model.named_modules():
        if type(module) in WRAPPERS:
            candidates.append((name, module))
        elif any(True for _ in module.parameters(recurse=False)):
            # Parameter-bearing, but not a type any wrapper reproduces: a
            # subclass, a bare-Parameter module, or something another transform
            # already wrapped (SMEAR installs MergedLinear during model
            # construction, which runs BEFORE this pass, so its targets are
            # invisible here). These used to be filtered out before the skip
            # accounting and so were reported NOWHERE - a broad profile could
            # miss a third of the model and still look like it covered
            # everything. Counted and named now.
            if entry.spec.matches(name):
                drop(f"unsupported:{type(module).__name__}", name)

    for name, module in candidates:
        if under_opaque(name):
            drop("opaque", name)
            continue
        if not entry.spec.matches(name):
            drop("unmatched", name)
            continue
        weight = getattr(module, "weight", None)
        if weight is None or not isinstance(weight, nn.Parameter):
            drop("unwrappable", name)
            continue
        if isinstance(weight, UninitializedParameter):
            # Lazy modules have no shape until the post-build dummy forward, and
            # ghostify runs before it. Same concession discover_targets makes.
            drop("lazy", name)
            continue
        if not weight.requires_grad:
            drop("frozen", name)
            continue
        if id(weight) in seen_ids:
            # Tied by reference. Ghosting one side and not the other unties them
            # silently, which is a different experiment than the one asked for,
            # so neither side is touched.
            drop("shared", name)
            continue
        if weight.numel() < MIN_TARGET_NUMEL:
            drop("too_small", name)
            continue
        if entry.skip_vocab and vocab is not None and vocab in tuple(weight.shape):
            drop("vocab", name)
            continue

        before = weight.numel()
        try:
            wrapper = WRAPPERS[type(module)](module, factory, tag=name)
        except ValueError:
            # Shape does not divide by d on both axes, or the module carries a
            # form the wrapper refuses (max_norm / sparse embeddings, grouped
            # convolutions). Reported, never silently skipped.
            drop("ineligible", name)
            continue

        seen_ids.add(id(weight))
        model.set_submodule(name, wrapper)
        targets.append((name, before, wrapper.expansion.real_numel))

    if not targets:
        raise ValueError(f"Ghost profile {profile!r} matched nothing. Skips: {skipped}")
    return GhostStats(profile, entry.rule, targets, skipped, missed)
