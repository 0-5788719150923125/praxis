import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, reduction="mean"):
        super().__init__()
        # self.alpha = alpha
        self.base_gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        embeddings: torch.Tensor,
        classifier: torch.Tensor,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
    ):
        logits = classifier(embeddings)
        shift_logits = logits[..., :-1, :]
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = labels[..., 1:].view(-1)
        ce_loss = F.cross_entropy(
            shift_logits, shift_labels, reduction="none", ignore_index=-100
        )
        pt = torch.exp(-ce_loss)
        # Detach can help the model stay stable during the training process.
        pt = pt.detach()
        gamma = self.base_gamma
        alpha = 1 / ((1 - pt) ** gamma).mean()
        # Alpha is a normalization factor that allows the training loss to be comparable to the cross-entropy loss.
        loss = alpha * (1 - pt) ** gamma * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
