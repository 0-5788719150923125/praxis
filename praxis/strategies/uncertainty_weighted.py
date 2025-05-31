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

    def __init__(self):
        super().__init__()
        # Use UninitializedParameter as a placeholder until first forward pass
        self.params = UninitializedParameter(requires_grad=True)

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
            # Handle negative losses (RL rewards) appropriately
            # Apply uncertainty weighting to absolute value to maintain mathematical consistency
            abs_loss = torch.abs(loss)
            weighted_abs_loss = 0.5 / (self.params[i] ** 2) * abs_loss
            
            # Preserve the sign to maintain optimization direction
            # Negative losses are reward signals that should be maximized (negative contribution)
            weighted_loss = torch.sign(loss) * weighted_abs_loss
            
            # Regularization term for the current loss
            reg_term = torch.log(1 + self.params[i] ** 2)

            # Add to total loss
            total_loss += weighted_loss + reg_term

        return total_loss
