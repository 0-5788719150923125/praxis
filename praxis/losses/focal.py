import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Original code found here:
    https://github.com/itakurah/Focal-loss-PyTorch/blob/main/focal_loss.py
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        classifier: torch.Tensor,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
    ):
        """Focal loss for multi-class classification."""
        shift_logits = logits[..., :-1, :]
        shift_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
        shift_labels = labels[..., 1:].reshape(-1)

        # Convert logits to probabilities with softmax
        probs = F.softmax(shift_logits, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(shift_labels, num_classes=logits.size(-1)).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal loss weight
        loss = self.alpha * focal_weight.unsqueeze(1) * ce_loss

        # Apply reductions
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
