from typing import List

import torch
from torch import Tensor, nn


class NaiveSummation(nn.Module):
    def forward(self, losses: List[Tensor]):
        return sum(losses)
