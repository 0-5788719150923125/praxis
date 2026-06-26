"""
ArcMixture: MixtureOfDepths with a per-recurrent-pass router bias.

Cyclic mixture-of-depths. Two axes, mirroring the ArcGLU idiom:

- Capacity is keyed to the *physical layer* index (``current_depth %
  num_layers``), so a given layer's sparsity is fixed across every recurrent
  pass. With the default ``arc`` schedule, odd layers route 75% sparse
  (capacity 0.25) and even layers run full - e.g. for num_layers=3 the 2nd
  layer (index 1) is the routed one on every pass.

- A zero-init ``nn.Embedding(num_passes, 1)`` adds a per-recurrent-pass scalar
  bias to the router logits (keyed to ``current_depth // num_layers``), so each
  time the layer is revisited it can shift its own routing threshold. This is
  the "adjust the bias at the recurrent step" mechanism shared with
  ArcAttention and ArcGLU, applied here to which tokens get routed.

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


class ArcMixture(MixtureOfDepths):
    """MixtureOfDepths keyed to physical layer, with a per-pass router bias.

    Capacity follows the ``arc`` schedule over ``num_layers`` (odd layers 75%
    sparse, even layers full) rather than over the flattened depth, so the same
    layer is the routed one on every recurrent pass. Each pass additionally
    gets its own zero-init additive bias on the router logits, so the model
    starts identical to MixtureOfDepths and specializes its routing threshold
    per pass over training.
    """

    # Depth-specialization diagnostics (see praxis.metrics.specialization),
    # averaged across ArcMixture routers and surfaced to the Dynamics tab.
    metric_descriptions = {
        "arc_router_specialization": {
            "description": (
                "Depth-specific fraction of the per-pass router bias "
                "(between-pass variance / total energy). 0 = every recurrent "
                "pass learned the same routing bias (collapsed, no benefit "
                "over a shared bias, and the zero-init case); rising = each "
                "pass is specializing its routing threshold."
            ),
            "chart": {
                "title": "Arc Depth Specialization",
                "y_label": "Specialized fraction",
                "y_scale": "linear",
                "group": "arc",
                "group_order": 30,
                "order": 13,
                "series_group": "arc_specialization",
                "series_label": "router bias",
            },
        },
    }

    def __init__(self, config: ConfigType, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)

        # Recurrent passes this layer will receive: ceil(depth / num_layers).
        self.num_passes = max(1, math.ceil(config.depth / self.num_layers))

        # Per-pass additive bias on the (scalar) router logit, applied after
        # the linear scorer. Zero-init -> starts identical to MixtureOfDepths.
        self.depth_router_bias = nn.Embedding(self.num_passes, 1)
        nn.init.zeros_(self.depth_router_bias.weight)

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

    def _compute_router_logits(self, inputs: Tensor, current_depth: int) -> Tensor:
        pass_idx = (current_depth // self.num_layers) % self.num_passes
        depth_idx = torch.tensor(pass_idx, device=inputs.device)
        return F.linear(inputs, self.weight, self.bias) + self.depth_router_bias(
            depth_idx
        )

    def training_metrics(self) -> dict:
        """Whether the per-pass router biases are specializing or collapsing."""
        from praxis.metrics.specialization import depth_dispersion

        out = {}
        disp = depth_dispersion(self.depth_router_bias.weight)
        if disp is not None:
            out["arc_router_specialization"] = disp["specialization"]
        return out
