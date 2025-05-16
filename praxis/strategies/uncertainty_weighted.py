from typing import List

import torch
from torch import Tensor, nn
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter


class UncertaintyWeighted(nn.Module, LazyModuleMixin):
    """
    Use homoscedastic uncertainty to balance multi-task losses.
    Adapted from:
    https://medium.com/@baicenxiao/strategies-for-balancing-multiple-loss-functions-in-deep-learning-e1a641e0bcc0
    """

    def __init__(self):
        super().__init__()
        # Use UninitializedParameter as a placeholder until first forward pass
        self.params = UninitializedParameter()

    def reset_parameters(self, num_params):
        # Initialize the parameters with the correct shape
        self.params.materialize(
            (num_params,), dtype=self.params.dtype, device=self.params.device
        )
        # Initialize with ones as in the original implementation
        with torch.no_grad():
            self.params.fill_(1.0)
        # Ensure parameters require gradients
        self.params.requires_grad_(True)

    def forward(self, losses: List[Tensor]) -> Tensor:
        # Check if parameters need to be initialized
        if self.has_uninitialized_params():
            # Initialize based on the length of the losses list
            self.reset_parameters(len(losses))

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
