from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.losses.reduction import weighted_reduce


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy with two optional extras: an anti-duplication
    penalty (``penalty_weight``) that up-weights tokens the model is about
    to predict equal to one already in the prompt, and per-token
    ``loss_weights`` for task-weighted training."""

    def __init__(self, penalty_weight: float = 0.0, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        input_ids: Tensor,
        loss_weights: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Cross entropy loss with optional duplication penalty and per-token weights.

        Args:
            logits: Predicted logits, already shifted to match labels.
            labels: Target labels.
            input_ids: Unshifted input token IDs (used for the dedup penalty).
            loss_weights: Optional per-token weight tensor matching ``labels``
                shape. See :func:`praxis.losses.reduction.weighted_reduce`.
        """
        shift_logits = logits.view(-1, logits.shape[-1])
        shift_labels = labels.view(-1)
        ce_loss = F.cross_entropy(
            shift_logits, shift_labels, reduction="none", ignore_index=-100
        )
        if self.penalty_weight != 0:
            token_output = torch.argmax(shift_logits, dim=1)
            duplicated_masks = (
                torch.eq(input_ids.view(-1), token_output.unsqueeze(-1))
                .any(dim=-1)
                .float()
            )
            ce_loss = ce_loss * (1 + duplicated_masks * self.penalty_weight)

        return weighted_reduce(ce_loss, labels=shift_labels, loss_weights=loss_weights)
