import torch
import torch.nn as nn
import torch.nn.functional as F


class MiLeLoss(nn.Module):
    """
    https://github.com/suu990901/LLaMA-MiLe-Loss
    """

    def __init__(self, gamma=1.0, reduction="mean"):
        super().__init__()
        self.base_gamma = gamma
        self.reduction = reduction
        self.sigma = 1

    def entropy(self, logits):
        # Detach can help the model stay stable during the training process.
        logits = logits.detach()
        probs = F.softmax(logits, dim=-1)
        epsilon = 1e-8
        probs = torch.clamp(probs, epsilon, 1.0)
        entropy = torch.sum(-1 * (probs * torch.log(probs)), dim=-1)
        return entropy

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", ignore_index=-100)
        pt = self.entropy(inputs)
        gamma = self.base_gamma
        alpha = 1.0 / (((self.sigma + pt) ** gamma).mean())
        # Alpha is a normalization factor that allows the training loss to be comparable to the cross-entropy loss.
        loss = alpha * ((self.sigma + pt) ** gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
