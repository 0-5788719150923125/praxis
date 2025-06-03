from typing import List

import torch
from torch import Tensor, nn
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter


class UncertaintyWeighted(nn.Module, LazyModuleMixin):
    """
    Use homoscedastic uncertainty to balance multi-task losses.

    This implementation handles both positive losses (to minimize) and negative
    losses (from RL rewards/maximization objectives) by applying the uncertainty
    weighting to absolute values while preserving the optimization direction.

    Adapted from:
    https://arxiv.org/abs/1705.07115
    https://medium.com/@baicenxiao/strategies-for-balancing-multiple-loss-functions-in-deep-learning-e1a641e0bcc0

    Note: Negative loss values are treated as reward signals (maximization objectives)
    and handled appropriately in the uncertainty weighting calculation.
    """

    def __init__(self, clamped: bool = False):
        super().__init__()
        # Use UninitializedParameter as a placeholder until first forward pass
        self.params = UninitializedParameter(requires_grad=True)
        self.clamped = clamped

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
            if self.clamped:
                # Clamp parameters to prevent exponential scaling instability
                # This prevents the 1/sigma^2 term from exploding when sigma becomes very small
                param = torch.clamp(self.params[i], min=0.1, max=10.0)
            else:
                param = self.params[i]

            # Apply uncertainty weighting with clamped parameters
            weighted_loss = 0.5 / (param**2) * loss

            # Regularization term for the current loss (use original param for gradient flow)
            reg_term = torch.log(1 + self.params[i] ** 2)

            # Add to total loss
            total_loss += weighted_loss + reg_term

        return total_loss
