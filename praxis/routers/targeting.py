"""PEFT-style target discovery for parameter-merging routers.

SMEAR/VEAR merge the parameters of ``num_experts`` copies of an ENTIRE decoder
block under one scalar per expert (praxis/routers/smear.py). That granularity is
the problem this module exists to fix: one coefficient decides the geometry of
the attention, the feedforward, both norms and both residual gates at once, and
the copies are paid for whether or not a given submodule has anything to gain
from being routed.

The alternative here is the one PEFT uses to decide where a LoRA goes: walk the
module tree, apply a named profile of include/exclude rules, and attach only
where the rules say to. A router built on this gets one coefficient row PER
DISCOVERED TARGET rather than one scalar for the block, and pays parameters only
for what it targeted.

Two exclusions are structural rather than stylistic, so they apply to every
profile and cannot be switched off:

  * ``MERGE_OPAQUE`` subtrees. A module that already routes its own parameters
    per token - PEER's product-key bank is the case here - gains nothing from a
    per-batch merge wrapped around it, and replicating it is the single most
    expensive thing the old design did. The flag is read off the class, the same
    "ask the registered class, not a hardcoded name list" idiom
    ``_wants_expert_bank`` uses in praxis/decoders/base.py.
  * Parameters shared BY REFERENCE. The long-term memory is tied across the
    block (praxis/decoders/base.py), and a tied tensor reached twice must be
    merged at most once or the two coefficient rows fight over one tensor.
    Deduplicated by ``id``, which also catches weight tying.

Lazy parameters are skipped, not errored: ``initialize_lazy_modules``
(praxis/utils/system.py) runs its dummy forward AFTER the model is built, so a
router constructed at build time cannot see the shapes of anything still
uninitialized. Everything lazy in this repo today is an activation
(Serpent/Servant), which is a per-feature nonlinearity rather than a geometry,
so nothing worth routing is lost. ``discover_targets`` reports what it skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch.nn as nn
from torch.nn.parameter import UninitializedParameter

# Below this many elements a factored delta costs MORE than storing the delta
# outright (a rank-r factorization of [out, in] holds r * (out + in) numbers), so
# small tensors - norm scales, residual gates, biases, kappa/mu - take the dense
# form. Shape-derived, not tuned.
DENSE_DELTA_MAX_NUMEL: int = 4096


@dataclass(frozen=True)
class TargetSpec:
    """Include/exclude rules for one named targeting profile.

    Patterns are regexes matched against a module's qualified name RELATIVE to
    the block root (``attn.qkv``, ``ffn_norm``, ...). The root itself matches as
    the empty string, which is how bare parameters hung directly off the block
    are reached.
    """

    include: Tuple[str, ...] = (r".*",)
    exclude: Tuple[str, ...] = ()
    # Skip any single parameter larger than this. Guards against a profile
    # accidentally replicating something enormous; None disables the check.
    max_param_numel: Optional[int] = None

    def matches(self, name: str) -> bool:
        if not any(re.fullmatch(p, name) for p in self.include):
            return False
        return not any(re.fullmatch(p, name) for p in self.exclude)


@dataclass
class TargetGroup:
    """One merge target: a module and the direct parameters it owns.

    ``params`` are fully-qualified names relative to the block, so they can be
    handed straight to ``torch.func.functional_call`` as state-dict keys.
    """

    name: str
    params: Tuple[str, ...]
    numel: int

    @property
    def label(self) -> str:
        """Metric-safe name. The root group has an empty qualname."""
        return (self.name or "root").replace(".", "_")


# Named profiles, registry-style. ``all`` is the principled default: merge
# everything that carries a geometry and does not already route itself. The
# narrower profiles exist to isolate where the gain (if any) comes from.
TARGET_PROFILES: Dict[str, TargetSpec] = {
    # Everything parameter-bearing that survives the structural exclusions
    # above - i.e. every module that does not already route itself. The default.
    "all": TargetSpec(),
    # Attention only - the sublayer the standard MoE literature does NOT make
    # sparse, so this is the honest test of whether it should be.
    "attn": TargetSpec(include=(r"attn(\..*)?",)),
    # Norms and residual gates only. Nearly free (a few hundred parameters per
    # expert) and the cheapest possible probe of whether per-module routing
    # buys anything at all.
    "gates": TargetSpec(include=(r".*_norm(\..*)?", r".*_res(\..*)?")),
    # Attention plus the norms/gates around it.
    "attn_gates": TargetSpec(
        include=(r"attn(\..*)?", r".*_norm(\..*)?", r".*_res(\..*)?")
    ),
}


def _opaque_prefixes(root: nn.Module) -> List[str]:
    """Qualnames of subtrees that opt out of merging via ``MERGE_OPAQUE``."""
    return [
        name
        for name, module in root.named_modules()
        if getattr(module, "MERGE_OPAQUE", False)
    ]


def _under(name: str, prefixes: List[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def discover_targets(
    root: nn.Module, spec: TargetSpec
) -> Tuple[List[TargetGroup], Dict[str, int]]:
    """Walk ``root`` and return the merge targets plus a skip tally.

    Only DIRECT parameters count toward a group (``recurse=False``), so a module
    that matches does not swallow its children's parameters - each child is
    discovered, and scored, on its own. That is what makes the coefficient
    granularity per-module rather than per-subtree.

    Returns ``(groups, skipped)`` where ``skipped`` counts parameters dropped by
    reason, so a caller can log what a profile actually excluded instead of
    silently targeting less than it thinks.
    """
    opaque = _opaque_prefixes(root)
    seen_ids: set = set()
    groups: List[TargetGroup] = []
    skipped: Dict[str, int] = {"opaque": 0, "lazy": 0, "shared": 0, "frozen": 0,
                               "oversized": 0, "unmatched": 0}

    for mod_name, module in root.named_modules():
        direct = list(module.named_parameters(recurse=False))
        if not direct:
            continue
        if _under(mod_name, opaque):
            skipped["opaque"] += len(direct)
            continue
        if not spec.matches(mod_name):
            skipped["unmatched"] += len(direct)
            continue

        names: List[str] = []
        numel = 0
        for pname, param in direct:
            if isinstance(param, UninitializedParameter):
                skipped["lazy"] += 1
                continue
            if not param.requires_grad:
                skipped["frozen"] += 1
                continue
            if id(param) in seen_ids:
                # Tied by reference (the shared long-term memory, weight tying).
                # Merging it twice would let two coefficient rows write the same
                # tensor; merging it once under one row is arbitrary, so drop it.
                skipped["shared"] += 1
                continue
            if (
                spec.max_param_numel is not None
                and param.numel() > spec.max_param_numel
            ):
                skipped["oversized"] += 1
                continue
            seen_ids.add(id(param))
            names.append(f"{mod_name}.{pname}" if mod_name else pname)
            numel += param.numel()

        if names:
            groups.append(TargetGroup(name=mod_name, params=tuple(names), numel=numel))

    return groups, skipped


def describe(groups: List[TargetGroup], skipped: Dict[str, int]) -> str:
    """One-line-per-target summary, for the build log and for tests."""
    total = sum(g.numel for g in groups)
    lines = [f"[SMEAR] {len(groups)} targets, {total:,} merged parameters"]
    for g in groups:
        lines.append(f"  {g.label:<24} {g.numel:>10,}  ({', '.join(g.params)})")
    drops = ", ".join(f"{k}={v}" for k, v in skipped.items() if v)
    if drops:
        lines.append(f"  skipped: {drops}")
    return "\n".join(lines)
