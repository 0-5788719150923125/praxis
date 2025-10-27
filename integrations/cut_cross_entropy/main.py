"""Cut Cross-Entropy loss integration implementation for Praxis."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from cut_cross_entropy import linear_cross_entropy
from torch import Tensor

from praxis.integrations.base import BaseIntegration


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
            embeddings: FULL UNSHIFTED embeddings from model (all sequence positions)
            classifier: Linear classifier layer
            labels: FULL UNSHIFTED labels (input_ids, not pre-shifted)
            **kwargs: Must contain 'input_ids' for unshifted targets

        Returns:
            Cut cross entropy loss value

        Note:
            This function uses shift=1 to let the library handle sequence shifting
            internally without materializing intermediate tensors. This is the key
            optimization that makes cut_cross_entropy memory-efficient.
        """
        # Use input_ids (unshifted) instead of labels (which may be pre-shifted)
        # The shift=1 parameter will handle the shifting internally
        targets = kwargs.get("input_ids", labels)

        # Apply pre-projection if the classifier has one (e.g., TiedClassifier with different hidden/embed sizes)
        if (
            hasattr(classifier, "pre_projection")
            and classifier.pre_projection is not None
        ):
            embeddings = classifier.pre_projection(embeddings)

        return linear_cross_entropy(
            embeddings,
            classifier.weight,
            targets,
            bias=getattr(classifier, "bias", None),
            impl="cce",
            shift=1,
            reduction="mean",
        )


def register_loss_functions():
    """Register cut_cross_entropy loss function with Praxis.

    This function is called by the integration loader to register
    the loss function with the LOSS_REGISTRY.
    """
    from praxis.losses import LOSS_REGISTRY

    LOSS_REGISTRY["cut_cross_entropy"] = CutCrossEntropyLoss

    return {
        "cut_cross_entropy": CutCrossEntropyLoss,
    }


def initialize(args, cache_dir, ckpt_path=None, truncated_hash=None):
    """Initialize cut_cross_entropy integration.

    This is called during Praxis initialization to set up the integration.
    """
    return {}


class Integration(BaseIntegration):
    """Cut Cross-Entropy integration class for Praxis."""

    def register_loss_functions(self) -> Dict[str, Any]:
        """Register loss functions provided by this integration."""
        return register_loss_functions()

    def initialize(
        self, args, cache_dir, ckpt_path=None, truncated_hash=None
    ) -> Dict[str, Any]:
        """Initialize the integration."""
        return initialize(args, cache_dir, ckpt_path, truncated_hash)
