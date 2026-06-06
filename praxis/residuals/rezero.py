from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from praxis.residuals.base import ResidualConnection


class ReZeroConnection(ResidualConnection):
    """ReZero (arXiv 1908.01188) with one gain per depth step.

    ``out = residual + alpha[depth] * branch``, alpha zero-init: the whole
    stack is the identity at entry and the model learns how much force each
    step applies - per depth, shared across sequence length. In a recurrent
    stack the same block reads a different gain at each loop step, so the
    loop's amplitude profile over depth is itself learned.
    """

    def __init__(self, dim: int, num_depths: int = 1, **kwargs: Any) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(max(1, int(num_depths))))

    def connect_depth(
        self, mix_h: Tensor, h_o: Tensor, beta: Tensor, current_depth: int = 0
    ) -> Tensor:
        idx = min(int(current_depth), self.alpha.numel() - 1)
        return mix_h + self.alpha[idx] * h_o
