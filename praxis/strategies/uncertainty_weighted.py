from typing import List

import torch
from torch import Tensor, nn


class UncertaintyWeighted(nn.Module):
    """
    Use homoscedastic uncertainty to balance multi-task losses.
    Adapted from:
    https://medium.com/@baicenxiao/strategies-for-balancing-multiple-loss-functions-in-deep-learning-e1a641e0bcc0
    """

    def __init__(self):
        super().__init__()
        self.params = None

    def forward(self, losses: List[Tensor]) -> Tensor:
        # Lazy-loading: initialize parameters based on number of losses
        if self.params is None:
            self.params = nn.Parameter(torch.ones(len(losses), requires_grad=True))

        total_loss = 0.0

        # Process each loss with its corresponding weight parameter
        for i, loss in enumerate(losses):
            # Weight for the current loss
            weighted_loss = 0.5 / (self.params[i] ** 2) * loss
            # Regularization term for the current loss
            reg_term = torch.log(1 + self.params[i] ** 2)

            # Add to total loss
            total_loss += weighted_loss + reg_term

        return total_loss
