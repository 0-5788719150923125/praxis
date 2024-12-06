import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class MiLeLoss(nn.Module):
    def __init__(self, gamma=1.0, reduction="mean"):
        super().__init__()
        self.base_gamma = gamma
        self.reduction = reduction
        self.loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        self.sigma = 1

    def entropy(self, logits):
        # Detach can help the model stay stable during the training process.
        logits = logits.detach()
        probs = F.softmax(logits, dim=-1)
        epsilon = 1e-8
        probs = torch.clamp(probs, epsilon, 1.0)
        entropy = torch.sum(-1 * (probs * torch.log(probs)), dim=-1)
        return entropy

    def forward(self, inputs, targets, global_steps):
        ce_loss = self.loss_fct(inputs, targets)
        pt = self.entropy(inputs)
        gamma = self.base_gamma
        alpha = 1.0 / (((self.sigma + pt) ** gamma).mean())
        torch.distributed.all_reduce(alpha)
        # Alpha is a normalization factor that allows the training loss to be comparable to the cross-entropy loss.
        alpha = alpha / dist.get_world_size()
        loss = alpha * ((self.sigma + pt) ** gamma) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
