import math
from typing import Any, Optional, TypeVar

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import UninitializedParameter

from praxis.activations import ACT2CLS
from praxis.dense.base import BaseDense

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class ArcGLU(BaseDense):
    """GLU with per-depth activation specialization.

    Mirrors GatedLinearMLP's up -> a*act(b) -> down structure, but each block
    owns a ModuleList sized to the number of recurrent passes *this* block
    will receive: ceil(depth / num_layers). The decoder routes depth via
    current_depth % num_layers, so each block sees current_depth values
    {i, i + num_layers, i + 2*num_layers, ...}; we index the activation list
    by current_depth // num_layers so each pass gets its own instance.
    """

    # Depth-specialization diagnostics (see praxis.metrics.specialization),
    # averaged across ArcGLU blocks and surfaced to the Dynamics tab. Only
    # parametric activations (Snake, PReLU, ...) register; parameter-free ones
    # (e.g. mish) emit nothing, since identical functions can't specialize.
    metric_descriptions = {
        "arc_act_specialization": {
            "description": (
                "Depth-specific fraction of the per-pass activation parameters "
                "(1 - ||mean||^2 / mean||row||^2). 0 = every recurrent pass "
                "learned identical activation params (collapsed); rising = "
                "passes specializing. Empty for parameter-free activations."
            ),
            "chart": {
                "title": "Arc Depth Specialization",
                "y_label": "Specialized fraction",
                "y_scale": "linear",
                "group": "arc",
                "group_order": 30,
                "order": 12,
                "series_group": "arc_specialization",
                "series_label": "activation",
            },
        },
        "arc_act_similarity": {
            "description": (
                "Mean pairwise cosine between the per-pass activation parameter "
                "vectors. ~1 = all passes converged to the same activation "
                "(collapsed); falling = passes diverging."
            ),
            "chart": {
                "title": "Arc Depth Similarity",
                "y_label": "Mean pairwise cosine",
                "y_scale": "linear",
                "group": "arc",
                "order": 22,
                "series_group": "arc_similarity",
                "series_label": "activation",
            },
        },
    }

    def __init__(
        self,
        config: ConfigType,
        activation: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        activation = activation or config.activation
        self.num_layers = max(1, config.num_layers)
        num_passes = max(1, math.ceil(config.depth / self.num_layers))

        down_size = int((4 / 3) * config.hidden_size)
        up_size = 2 * down_size

        self.up: nn.Linear = nn.Linear(config.hidden_size, up_size)
        self.act: nn.ModuleList = nn.ModuleList(
            [ACT2CLS[activation](*args, **kwargs) for _ in range(num_passes)]
        )
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)
        self.down: nn.Linear = nn.Linear(down_size, config.hidden_size)

    def forward(
        self,
        inputs: Tensor,
        current_depth: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        a, b = self.up(inputs).chunk(2, dim=-1)
        act = self.act[(current_depth // self.num_layers) % len(self.act)]
        return self.down(self.dropout(a * act(b)))

    def training_metrics(self) -> dict:
        """Whether the per-pass activations are specializing or collapsing.

        Stacks each pass's flattened activation params into a [P, n_params]
        matrix and reports its dispersion. Empty when there's nothing to
        measure (single pass, or parameter-free activations).
        """
        from praxis.metrics.specialization import depth_dispersion

        if len(self.act) < 2:
            return {}
        rows = []
        for act in self.act:
            params = list(act.parameters())
            if not params or any(isinstance(p, UninitializedParameter) for p in params):
                return {}
            rows.append(torch.cat([p.detach().reshape(-1) for p in params]))

        disp = depth_dispersion(torch.stack(rows, dim=0))
        if disp is None:
            return {}
        return {
            "arc_act_specialization": disp["specialization"],
            "arc_act_similarity": disp["similarity"],
        }
