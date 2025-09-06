from typing import List

import torch
from torch import Tensor, nn


class RealTime(nn.Module):
    """
    In this approach, the loss function for each task is adjusted to always equal 1,
    ensuring consistent magnitudes. Due to the presence of the detach operation, its
    gradient is not always zero. Simply put, we can treat the detached loss as a constant.
    Thus, the final result dynamically adjusts the gradient proportions using the latest
    loss value.
    https://medium.com/@baicenxiao/strategies-for-balancing-multiple-loss-functions-in-deep-learning-e1a641e0bcc0
    """

    def forward(self, losses: List[Tensor]):
        # Normalize losses and combine them
        return sum([loss / loss.detach() for loss in losses])
