import torch
import torch.nn as nn
import torch.nn.functional as F
from cut_cross_entropy import linear_cross_entropy


class CutCrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(
        self,
        embeddings: torch.Tensor,
        classifier: torch.Tensor,
        labels: torch.Tensor,
        *args,
        **kwargs,
    ):
        return linear_cross_entropy(
            embeddings,
            classifier.weight,
            labels,
            bias=classifier.bias,
            impl="torch_compile",
            shift=1,
            reduction="mean",
        )
