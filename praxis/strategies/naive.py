from typing import List

import torch
from torch import Tensor, nn


class NaiveSummation(nn.Module):
    """Sum all per-task losses with equal weight. The baseline strategy:
    works fine when tasks are well-scaled, breaks when they aren't."""

    def forward(self, losses: List[Tensor]):
        return sum(losses)
