from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossEntropyLoss(nn.Module):
    def __init__(self, penalty_weight: float = 0.0, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        input_ids: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """
        Calculate the cross entropy loss with optional penalty for duplicated tokens.

        Args:
            logits: Predicted logits
            labels: Target labels
            input_ids: Input token IDs

        Returns:
            Cross entropy loss with optional duplication penalty
        """
        shift_logits = logits.view(-1, logits.shape[-1])
        shift_labels = labels.view(-1)
        ce_loss = F.cross_entropy(
            shift_logits, shift_labels, reduction="none", ignore_index=-100
        )
        if self.penalty_weight == 0:
            return ce_loss.mean()
        token_output = torch.argmax(shift_logits, dim=1)
        duplicated_masks = (
            torch.eq(input_ids.view(-1), token_output.unsqueeze(-1)).any(dim=-1).float()
        )
        loss = ce_loss * (1 + duplicated_masks * self.penalty_weight)
        return loss.mean()
