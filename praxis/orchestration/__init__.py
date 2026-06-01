"""Distributed remote-expert pooling for Praxis.

A pool of tiny experts (local, Ray, Hivemind, or browser peers over GUN) behind
one mixing layer. Each expert owns its weights + a local projection and does
detached, layer-wise (Mono-Forward) updates, so no gradient crosses the network:
the training forward is non-blocking and the inference forward is stochastically
sampled and mixed. This is the Python side of the in-browser swarm (the JS twin
is praxis/web/src/js/swarm.js); see next/world_models.md for the vision and the
rank-priced connection.

Pluggable by registry, mirroring the rest of Praxis:

* ``EXPERT_REGISTRY`` - expert *block* types a local expert can wrap (the dense
  block zoo, kept for backward compat with the integrations layer).
* ``MIXING_REGISTRY`` - how the pool combines experts at inference; the chosen
  orchestration profile names one: mean / vote / sample / standing-wave.

We are building an in-house alternative to Hivemind; the ``RemoteExpert``
interface is the transport seam where a Hivemind/Ray/GUN backend plugs in.
"""

from typing import Any, Dict, List, Optional, Type

from torch import nn

from praxis.dense import DENSE_REGISTRY
from praxis.layers import LocalLayer, RemoteLayer
from praxis.orchestration.base import LocalExpert, RemoteExpert
from praxis.orchestration.mixing import (
    MIXING_DESCRIPTIONS,
    MIXING_REGISTRY,
    build_mixer,
)
from praxis.orchestration.pool import ExpertPool
from praxis.orchestration.sidecar import SidecarExpert, SidecarManager

# Expert block types a LocalExpert can wrap (backward compatible: the Hivemind
# integration resolves expert classes from here).
EXPERT_REGISTRY: Dict[str, Type[nn.Module]] = {**DENSE_REGISTRY}

# Named orchestration profiles. A single ``--orchestration-type`` flag selects
# one (mirrors ``--memory-type``); each profile bundles whether to spawn the
# backend sidecar, the starter expert count, and the inference mixing strategy,
# so new variants are registry entries, not new CLI knobs. ``None`` disables the
# pool entirely (the common case).
ORCHESTRATION_REGISTRY: Dict[str, Optional[dict]] = {
    "none": None,
    # In-process pool of tiny experts, joinable from the web Stage tab. The
    # `sidecar_*` variants additionally spawn the Node sidecar of browser-math
    # experts as extra peers.
    "swarm": dict(sidecar=False, init_experts=4, mixing="vote"),
    "swarm_mean": dict(sidecar=False, init_experts=4, mixing="mean"),
    "swarm_wave": dict(sidecar=False, init_experts=4, mixing="wave"),
    "swarm_sidecar": dict(sidecar=True, init_experts=4, mixing="vote"),
    # No baseline experts - mix only whatever joins from the frontend.
    "frontend_only": dict(sidecar=False, init_experts=0, mixing="vote"),
}

ORCHESTRATION_DESCRIPTIONS: Dict[str, str] = {
    "none": "Disabled. No remote-expert pool.",
    "swarm": "In-process pool of 4 tiny experts; CALM-style expert vote.",
    "swarm_mean": "In-process pool of 4 experts; plain mean of expert outputs.",
    "swarm_wave": "In-process pool of 4 experts; standing-wave mix over peers.",
    "swarm_sidecar": "In-process pool of 4 + a Node sidecar of browser-math experts.",
    "frontend_only": "No baseline experts; mix only frontend-joined experts (vote).",
}


def get_orchestration_profile(name: str) -> Optional[dict]:
    """Resolve an ``--orchestration-type`` name to its profile spec (or None)."""
    if name not in ORCHESTRATION_REGISTRY:
        raise KeyError(
            f"unknown orchestration_type {name!r}; choices: "
            f"{sorted(ORCHESTRATION_REGISTRY)}"
        )
    return ORCHESTRATION_REGISTRY[name]


def build_pool(
    experts: Optional[List[RemoteExpert]] = None,
    mixing: str = "mean",
    sample_size: Optional[int] = None,
) -> ExpertPool:
    """Construct an :class:`ExpertPool` (convenience wrapper)."""
    return ExpertPool(experts=experts, mixing=mixing, sample_size=sample_size)


__all__ = [
    "EXPERT_REGISTRY",
    "ORCHESTRATION_REGISTRY",
    "ORCHESTRATION_DESCRIPTIONS",
    "get_orchestration_profile",
    "MIXING_REGISTRY",
    "MIXING_DESCRIPTIONS",
    "RemoteExpert",
    "LocalExpert",
    "ExpertPool",
    "SidecarExpert",
    "SidecarManager",
    "build_mixer",
    "build_pool",
]
