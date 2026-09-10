"""Regression terms, for objectives whose target is not a token."""

from typing import Any, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.losses.reduction import weighted_reduce


class MeanSquaredErrorLoss(nn.Module):
    """Squared error against a continuous target.

    MTP's patch path predicts patch embeddings rather than tokens, so its
    term is a regression. Same call shape as the token-level terms -
    predictions, targets, optional per-position weights - so a producer can
    swap one for the other without changing its call site.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        loss_weights: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        per_position = F.mse_loss(preds, targets, reduction="none").mean(-1)
        return weighted_reduce(per_position, loss_weights=loss_weights)


class SmoothL1Loss(nn.Module):
    """Huber regression against a continuous target.

    The mono-forward decoder's latent goodness scores a layer's projection
    against the next step of the encoder's own stream, where a squared error
    would let one badly-placed position dominate the cut.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        loss_weights: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        per_position = F.smooth_l1_loss(preds, targets, reduction="none").mean(-1)
        return weighted_reduce(per_position, loss_weights=loss_weights)
