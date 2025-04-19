import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        classifier: torch.Tensor,
        labels: torch.Tensor,
        *args,
        **kwargs,
    ):
        shift_logits = logits[..., :-1, :]
        shift_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
        shift_labels = labels[..., 1:].reshape(-1).unsqueeze(1)

        # Convert logits to probabilities with softmax
        log_probs = F.log_softmax(shift_logits, dim=1)

        # Get the log probability and probability of the correct class
        log_p_t = log_probs.gather(1, shift_labels)
        p_t = torch.exp(log_p_t)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal loss weight to all classes
        loss = -self.alpha * focal_weight * log_p_t

        # Apply reductions
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
