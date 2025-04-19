import torch
import torch.nn as nn
import torch.nn.functional as F
from cut_cross_entropy import linear_cross_entropy


class CutCrossEntropyLoss(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        classifier: torch.Tensor,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
    ):
        loss = linear_cross_entropy(
            embeddings,
            classifier.weight,
            labels,
            bias=classifier.bias,
            impl="torch_compile",
            shift=1,
            reduction="mean",
        )
        return loss
