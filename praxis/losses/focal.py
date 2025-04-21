from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    """

    def __init__(
        self,
        alpha: Union[float, torch.Tensor] = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Handle alpha as a scalar or a vector
        if isinstance(alpha, (float, int)):
            self.alpha_scalar = float(alpha)
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha_vector", alpha)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        # 1. Shift inputs for next-token prediction and reshape
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1).unsqueeze(1)

        # 2. Calculate probabilities and focal loss components
        log_probs = F.log_softmax(shift_logits, dim=1)  # Shape: (N, V)

        # Gather requires index shape (N, 1) for dim=1 -> squeeze back to (N,)
        log_p_t = log_probs.gather(1, shift_labels)
        p_t = torch.exp(log_p_t)
        focal_weight = (1 - p_t) ** self.gamma  # Shape: (N,)

        # 3. Apply alpha - dynamically determine which alpha to use
        if hasattr(self, "alpha_vector"):
            alpha_factor = self.alpha_vector[shift_labels]  # Shape: (N,)
        else:
            alpha_factor = self.alpha_scalar

        # 4. Compute loss per token
        loss = -alpha_factor * focal_weight * log_p_t  # Shape: (N,)

        # 5. Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        # else: # reduction == "none"
        return loss  # Shape (N,)
