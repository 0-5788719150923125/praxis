from typing import Any

import torch.nn as nn
from torch import Tensor


class BaseDense(nn.Module):
    """Shared contract for feedforward/"dense" modules in decoder blocks.

    Subclasses must accept `current_depth` so depth-aware variants (e.g.
    ArcGLU) can index per-depth parameters. Depth-agnostic variants simply
    ignore it.
    """

    def forward(
        self,
        inputs: Tensor,
        current_depth: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        raise NotImplementedError
