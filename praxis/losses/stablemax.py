from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis.losses.reduction import weighted_reduce


class StableMaxCrossEntropyLoss(nn.Module):
    """
    From "Grokking at the Edge of Numerical Stability":
    https://arxiv.org/abs/2501.04697
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weights: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ):
        shift_logits = logits.view(-1, logits.shape[-1])
        shift_labels = labels.view(-1).to(torch.int64)
        logprobs = log_stablemax(shift_logits.to(torch.float64), dim=-1)
        prediction_logprobs = torch.gather(
            logprobs, index=shift_labels[:, None], dim=-1
        ).to(torch.float64)
        per_token = -prediction_logprobs.squeeze(-1)
        return weighted_reduce(
            per_token, labels=shift_labels, loss_weights=loss_weights
        )


def s(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))
