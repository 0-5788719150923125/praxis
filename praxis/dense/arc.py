import math
from typing import Any, Optional, TypeVar

import torch.nn as nn
from torch import Tensor

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
