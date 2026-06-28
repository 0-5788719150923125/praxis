"""
ArcMixture: MixtureOfDepths with a per-recurrent-pass router weight delta.

Cyclic mixture-of-depths. Two axes, mirroring the ArcGLU idiom:

- Capacity is keyed to the *physical layer* index (``current_depth %
  num_layers``), so a given layer's sparsity is fixed across every recurrent
  pass. With the default ``arc`` schedule, odd layers route 75% sparse
  (capacity 0.25) and even layers run full - e.g. for num_layers=3 the 2nd
  layer (index 1) is the routed one on every pass.

- Each recurrent pass gets its own low-rank additive delta on the router's
  weight *vector* (keyed to ``current_depth // num_layers``), so the routing
  direction itself - not just a uniform threshold - can specialize per pass.
  A uniform scalar bias is rank-invariant to the top-k and so can never change
  *which* tokens route; only a delta on the weight can re-rank them. The delta
  is a shared ``[rank, hidden]`` basis (LoRA "A") with per-pass coefficients
  ``nn.Embedding(num_passes, rank)`` (LoRA "B", zero-init), so the model starts
  identical to MixtureOfDepths and the passes diverge over training. The base
  ``nn.Linear`` bias is dropped: a per-token-uniform scalar adds nothing the
  delta cannot, and it competed with the per-pass term for the same calibration.

The decoder routes physical block ``i`` whenever ``current_depth % num_layers
== i`` (base controller, round-robin), so block ``i`` sees ``current_depth``
values ``{i, i + num_layers, ...}``; we recover the layer index with ``%`` and
the pass index with ``//``.
"""

import math
from typing import Any, List, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.routers.mixture_of_depths import MixtureOfDepths
from praxis.utils import generate_alternating_values

ConfigType = TypeVar("ConfigType", bound="AutoConfig")

# Rank of the per-pass weight delta. Small relative to hidden_size: the deltas
# only need to nudge the routing direction, and a low rank keeps the per-pass
# coefficient count tiny while sharing a common basis across passes.
_DELTA_RANK = 8


class ArcMixture(MixtureOfDepths):
    """MixtureOfDepths keyed to physical layer, with a per-pass weight delta.

    Capacity follows the ``arc`` schedule over ``num_layers`` (odd layers 75%
    sparse, even layers full) rather than over the flattened depth, so the same
    layer is the routed one on every recurrent pass. Each pass additionally
    gets its own low-rank additive delta on the router weight vector, so the
    routing *direction* (not just a uniform threshold) specializes per pass.
    Zero-init coefficients mean the model starts identical to MixtureOfDepths.
    """

    # Depth-specialization diagnostics (see praxis.metrics.specialization),
    # averaged across ArcMixture routers and surfaced to the Dynamics tab.
    metric_descriptions = {
        "arc_router_specialization": {
            "description": (
                "Depth-specific fraction of the per-pass router weight delta "
                "(between-pass variance / total energy). 0 = every recurrent "
                "pass learned the same routing direction (collapsed, no benefit "
                "over a shared router, and the zero-init case); rising = each "
                "pass is specializing which tokens it routes."
            ),
            "chart": {
                "title": "Arc Depth Specialization",
                "y_label": "Specialized fraction",
                "y_scale": "linear",
                "group": "arc",
                "group_order": 30,
                "order": 13,
                "series_group": "arc_specialization",
                "series_label": "router delta",
            },
        },
    }

    def __init__(self, config: ConfigType, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)

        # Drop the shared per-token-uniform scalar bias: it adds nothing the
        # per-pass delta cannot, and it competed for the same calibration.
        self.bias = None

        # Recurrent passes this layer will receive: ceil(depth / num_layers).
        self.num_passes = max(1, math.ceil(config.depth / self.num_layers))

        # Low-rank per-pass delta on the router weight vector [1, hidden]:
        #   delta_w[pass] = coef[pass] @ basis            ([rank]@[rank,hidden])
        # Shared ``basis`` (LoRA "A", random), per-pass ``coef`` (LoRA "B",
        # zero-init) -> delta is exactly 0 at init, so logits match base MoD.
        rank = min(_DELTA_RANK, config.hidden_size)
        self.delta_rank = rank
        self.delta_basis = nn.Parameter(
            torch.empty(rank, config.hidden_size).normal_(std=config.hidden_size**-0.5)
        )
        self.delta_coef = nn.Embedding(self.num_passes, rank)
        nn.init.zeros_(self.delta_coef.weight)

    def _build_capacities(self, config: ConfigType, layout: str) -> List[float]:
        # Key capacity to the physical layer index (length num_layers), not the
        # flattened depth: odd layers route 75% sparse, even layers run full.
        self.num_layers = max(1, config.num_layers)
        return generate_alternating_values(
            size=self.num_layers, interval=1, capacity=0.25
        )

    def _capacity_for(self, current_depth: int) -> float:
        # Physical block i is visited whenever current_depth % num_layers == i.
        return self.capacities[current_depth % self.num_layers]

    def _pass_delta(self, pass_idx: int, device) -> Tensor:
        """Per-pass additive delta on the router weight vector, shape [1, hidden]."""
        coef = self.delta_coef(torch.tensor(pass_idx, device=device))  # [rank]
        return (coef @ self.delta_basis).unsqueeze(0)  # [1, hidden]

    def _compute_router_logits(self, inputs: Tensor, current_depth: int) -> Tensor:
        pass_idx = (current_depth // self.num_layers) % self.num_passes
        weight = self.weight + self._pass_delta(pass_idx, inputs.device)
        return F.linear(inputs, weight, self.bias)

    def _pass_deltas(self) -> Tensor:
        """All per-pass weight deltas stacked, shape [num_passes, hidden]."""
        return self.delta_coef.weight @ self.delta_basis

    def training_metrics(self) -> dict:
        """Whether the per-pass router deltas are specializing or collapsing."""
        from praxis.metrics.specialization import depth_dispersion

        out = {}
        disp = depth_dispersion(self._pass_deltas())
        if disp is not None:
            out["arc_router_specialization"] = disp["specialization"]
        return out
