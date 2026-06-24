"""
ArcMixture: MixtureOfDepths with a per-recurrent-depth router bias.

Adds an nn.Embedding(depth, 1) additive bias to the router logits, looked up
via current_depth, so each recurrent depth pass can shift its own routing
threshold. Zero-initialized: the router starts identical to MixtureOfDepths
and gradually specializes per depth.

This is the routing analog of ArcAttention/ArcGLU, which add per-depth biases
to the attention projections and per-pass activations respectively - "adjust
the bias at the recurrent step", applied to which tokens get routed.
"""

from typing import Any, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.routers.mixture_of_depths import MixtureOfDepths

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class ArcMixture(MixtureOfDepths):
    """MixtureOfDepths that conditions the router on recurrent depth.

    Each recurrent depth pass gets its own additive scalar bias on the router
    logits via an nn.Embedding lookup. Zero-initialized so the model starts
    identical to MixtureOfDepths and gradually specializes its routing per
    depth - the same "adjust the bias at the recurrent step" mechanism used by
    ArcAttention and ArcGLU.

    Defaults to the ``arc`` layout: full capacity on even layers and 75%
    sparsity (capacity 0.25) on odd layers.
    """

    # Depth-specialization diagnostics (see praxis.metrics.specialization),
    # averaged across ArcMixture routers and surfaced to the Dynamics tab.
    metric_descriptions = {
        "arc_router_specialization": {
            "description": (
                "Depth-specific fraction of the per-depth router bias "
                "(between-depth variance / total energy). 0 = every recurrent "
                "depth learned the same routing bias (collapsed, no benefit "
                "over a shared bias, and the zero-init case); rising = each "
                "depth is specializing its routing threshold."
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

    def __init__(
        self, config: ConfigType, layout: str = "arc", *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(config, layout=layout, *args, **kwargs)

        self.depth = config.depth

        # Per-depth additive bias on the (scalar) router logit, applied after
        # the linear scorer. Zero-init -> starts identical to MixtureOfDepths.
        self.depth_router_bias = nn.Embedding(self.depth, 1)
        nn.init.zeros_(self.depth_router_bias.weight)

    def _compute_router_logits(self, inputs: Tensor, current_depth: int) -> Tensor:
        depth_idx = torch.tensor(current_depth, device=inputs.device)
        return F.linear(inputs, self.weight, self.bias) + self.depth_router_bias(
            depth_idx
        )

    def training_metrics(self) -> dict:
        """Whether the per-depth router biases are specializing or collapsing."""
        from praxis.metrics.specialization import depth_dispersion

        out = {}
        disp = depth_dispersion(self.depth_router_bias.weight)
        if disp is not None:
            out["arc_router_specialization"] = disp["specialization"]
        return out
