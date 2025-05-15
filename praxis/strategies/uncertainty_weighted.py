import torch
from torch import Tensor, nn


class UncertaintyWeighted(nn.Module):
    """
    Use homoscedastic uncertainty to balance single-task losses.
    Adapted from:
    https://medium.com/@baicenxiao/strategies-for-balancing-multiple-loss-functions-in-deep-learning-e1a641e0bcc0
    """

    def __init__(self):
        super().__init__()
        # Initialize weight parameters for each loss (similar to blog post)
        self.params = nn.Parameter(torch.ones(2, requires_grad=True))

    def forward(self, main_loss: Tensor, aux_loss: Tensor):
        # Weight for main loss
        weighted_main = 0.5 / (self.params[0] ** 2) * main_loss
        # Regularization for main loss
        reg_main = torch.log(1 + self.params[0] ** 2)

        # Weight for auxiliary loss
        weighted_aux = 0.5 / (self.params[1] ** 2) * aux_loss
        # Regularization for auxiliary loss
        reg_aux = torch.log(1 + self.params[1] ** 2)

        # Calculate final weighted loss
        loss = weighted_main + reg_main + weighted_aux + reg_aux

        return loss
