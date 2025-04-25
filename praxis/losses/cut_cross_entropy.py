from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from cut_cross_entropy import linear_cross_entropy
from torch import Tensor


class CutCrossEntropyLoss(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(
        self,
        embeddings: Tensor,
        classifier: nn.Linear,
        labels: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """
        Calculate the cut cross entropy loss.

        Args:
            embeddings: Input embeddings from model
            classifier: Linear classifier layer
            labels: Target labels

        Returns:
            Cut cross entropy loss value
        """
        return linear_cross_entropy(
            embeddings,
            classifier.weight,
            labels,
            bias=classifier.bias,
            impl="torch_compile",
            shift=1,
            reduction="mean",
        )
