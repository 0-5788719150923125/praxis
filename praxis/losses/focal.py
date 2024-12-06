import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, reduction="mean"):
        super().__init__()
        # self.alpha = alpha
        self.base_gamma = gamma
        self.reduction = reduction
        self.loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    def forward(self, inputs, targets, global_steps):
        ce_loss = self.loss_fct(inputs, targets)
        pt = torch.exp(-ce_loss)
        # Detach can help the model stay stable during the training process.
        pt = pt.detach()
        gamma = self.base_gamma
        alpha = 1 / ((1 - pt) ** gamma).mean()
        torch.distributed.all_reduce(alpha)
        alpha = alpha / dist.get_world_size()
        # Alpha is a normalization factor that allows the training loss to be comparable to the cross-entropy loss.
        loss = alpha * (1 - pt) ** gamma * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
