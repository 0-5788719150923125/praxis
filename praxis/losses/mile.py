import torch
import torch.nn as nn
import torch.nn.functional as F


class MiLeLoss(nn.Module):
    """
    An implementation of [Mi]tigating the bias of [le]arning difficulties with tokens.
    https://github.com/suu990901/LLaMA-MiLe-Loss
    Probably better than standard cross-entropy, but uses more VRAM, so it's not the default in Praxis today.
    """

    def __init__(self, gamma=1.0, reduction="mean"):
        super().__init__()
        self.base_gamma = gamma
        self.reduction = reduction
        self.sigma = 1
        self.ema_decay = 0.9
        self.register_buffer("ema_alpha", torch.tensor(1.0))

    def entropy(self, logits):
        # Detach can help the model stay stable during the training process.
        logits = logits.detach()
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        ce_loss = F.cross_entropy(logits, labels, reduction="none", ignore_index=-100)
        pt = self.entropy(logits)
        gamma = self.base_gamma
        # Alpha is a normalization factor that allows the training loss to be comparable to the cross-entropy loss.
        scale = (self.sigma + pt) ** gamma
        current_alpha = 1.0 / (scale.mean())
        # Update EMA alpha
        with torch.no_grad():
            self.ema_alpha = (
                self.ema_decay * self.ema_alpha + (1 - self.ema_decay) * current_alpha
            )
        # Use EMA alpha for loss calculation; simulates the multi-GPU normalization used in the original code
        loss = self.ema_alpha * scale * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
