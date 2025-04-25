import torch
import torch.nn as nn
import torch.nn.functional as F


class StableMaxCrossEntropyLoss(nn.Module):
    """
    From "Grokking at the Edge of Numerical Stability":
    https://arxiv.org/abs/2501.04697
    """

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *args,
        **kwargs,
    ):
        shift_logits = logits.view(-1, logits.shape[-1])
        shift_labels = labels.view(-1).to(torch.int64)
        logprobs = log_stablemax(shift_logits.to(torch.float64), dim=-1)
        prediction_logprobs = torch.gather(
            logprobs, index=shift_labels[:, None], dim=-1
        ).to(torch.float64)
        loss = -torch.mean(prediction_logprobs)
        return loss


def s(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))
