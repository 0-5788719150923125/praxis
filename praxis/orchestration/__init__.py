"""Distributed remote-expert pooling for Praxis.

A pool of tiny experts (local, Ray, Hivemind, or browser peers over GUN) behind
one mixing layer. Each expert owns its weights + a local projection and does
detached, layer-wise (Mono-Forward) updates, so no gradient crosses the network:
the training forward is non-blocking and the inference forward is stochastically
sampled and mixed. This is the Python side of the in-browser swarm (the JS twin
is praxis/web/src/js/swarm.js); see next/world_models.md for the vision and the
rank-priced connection.

Pluggable by registry, mirroring the rest of Praxis:

* the ``dense`` registry - expert *block* types a local expert can wrap.
* the ``mixing`` registry - how the pool combines experts at inference; the chosen
  orchestration profile names one: mean / vote / sample / standing-wave.

We are building an in-house alternative to Hivemind; the ``RemoteExpert``
interface is the transport seam where a Hivemind/Ray/GUN backend plugs in.
"""

from typing import Any, List, Optional

from praxis import registry
from praxis.layers import LocalLayer, RemoteLayer
from praxis.orchestration.base import LocalExpert, RemoteExpert
from praxis.orchestration.mixing import build_mixer
from praxis.orchestration.pool import ExpertPool
from praxis.orchestration.sidecar import SidecarExpert, SidecarManager
from praxis.registry import Entry

registry.declare(
    "orchestration",
    title="Remote-expert orchestration",
    doc=(
        (
            "Pool profiles for the distributed swarm. Each profile bundles whether to "
            "spawn the backend sidecar, the starter expert count and the inference mixing "
            "strategy, so new variants are entries, not new flags. The expert block types "
            "a pool member can wrap are the feedforward experts (the ``dense`` registry)."
        )
    ),
    entries={
        "none": Entry(
            None,
            "Disabled. No remote-expert pool, the common case.",
        ),
        "swarm": Entry(
            dict(sidecar=False, init_experts=4, mixing="vote"),
            (
                "An in-process pool of 4 tiny experts, joinable from the web Stage "
                "tab, mixed at inference by the CALM-style expert vote."
            ),
        ),
        "swarm_mean": Entry(
            dict(sidecar=False, init_experts=4, mixing="mean"),
            "swarm mixed by a plain mean of the expert outputs instead of a vote.",
        ),
        "swarm_wave": Entry(
            dict(sidecar=False, init_experts=4, mixing="wave"),
            (
                "swarm mixed by a standing wave over the peer index, so peers compose "
                "by interference rather than a flat average."
            ),
        ),
        "swarm_sidecar": Entry(
            dict(sidecar=True, init_experts=4, mixing="vote"),
            (
                "swarm plus a spawned Node sidecar of browser-math experts, which join "
                "the pool as extra peers."
            ),
        ),
        "frontend_only": Entry(
            dict(sidecar=False, init_experts=0, mixing="vote"),
            (
                "No baseline experts: the pool mixes, by vote, only the experts that "
                "join from the frontend."
            ),
        ),
    },
)


def get_orchestration_profile(name: str) -> Optional[dict]:
    """Resolve an ``--orchestration-type`` name to its profile spec (or None)."""
    if name not in registry.namespace("orchestration"):
        raise KeyError(
            f"unknown orchestration_type {name!r}; choices: "
            f"{sorted(registry.namespace("orchestration"))}"
        )
    return registry.lookup("orchestration", name)


def build_pool(
    experts: Optional[List[RemoteExpert]] = None,
    mixing: str = "mean",
    sample_size: Optional[int] = None,
) -> ExpertPool:
    """Construct an :class:`ExpertPool` (convenience wrapper)."""
    return ExpertPool(experts=experts, mixing=mixing, sample_size=sample_size)


__all__ = [
    "get_orchestration_profile",
    "RemoteExpert",
    "LocalExpert",
    "ExpertPool",
    "SidecarExpert",
    "SidecarManager",
    "build_mixer",
    "build_pool",
]
