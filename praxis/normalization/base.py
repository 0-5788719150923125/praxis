"""Base normalization class that provides a no-op implementation."""

from abc import ABC
from typing import Any

import torch
import torch.nn as nn


class BaseNorm(nn.Module, ABC):
    """
    Base normalization class that provides a no-op implementation.

    Other normalization classes should inherit from this and override
    the forward method to implement their specific normalization logic.
    """

    def __init__(
        self,
        normalized_shape: Any,
        eps: float = 1e-05,
        pre_norm: bool = True,
        post_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.pre_norm = pre_norm
        self.post_norm = post_norm

    def forward(self, input: torch.Tensor, mode: str = "direct") -> torch.Tensor:
        """
        Apply normalization based on mode.

        Args:
            input: Input tensor
            mode: "pre", "post", "both", "none", or "direct"

        Returns:
            Normalized tensor (identity for base implementation)
        """
        return input


# Alias for backward compatibility
NoNorm = BaseNorm