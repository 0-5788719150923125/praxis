from typing import Any, TypeVar

import torch
import torch.nn as nn

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class NoNorm(nn.Module):
    """Identity normalization that applies no transformation."""

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
            Normalized tensor (identity for NoNormalization)
        """
        return input


class LayerNorm(nn.LayerNorm):
    """LayerNorm with configurable pre/post norm behavior."""

    def __init__(
        self,
        normalized_shape: Any,
        eps: float = 1e-05,
        pre_norm: bool = True,
        post_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(normalized_shape, eps=eps, **kwargs)
        self.pre_norm = pre_norm
        self.post_norm = post_norm

    def forward(self, input: torch.Tensor, mode: str = "direct") -> torch.Tensor:
        """
        Apply normalization based on mode.

        Args:
            input: Input tensor
            mode: "pre", "post", "both", "none", or "direct"

        Returns:
            Normalized tensor
        """
        if mode == "pre" and self.pre_norm:
            return super().forward(input)
        elif mode == "post" and self.post_norm:
            return super().forward(input)
        elif mode == "both" and (self.pre_norm or self.post_norm):
            return super().forward(input)
        elif mode == "direct":
            return super().forward(input)
        elif mode == "none":
            return input
        else:
            # No normalization applied for this mode/config combination
            return input


class RMSNorm(nn.RMSNorm):
    """RMSNorm with configurable pre/post norm behavior."""

    def __init__(
        self,
        normalized_shape: Any,
        eps: float = 1e-05,
        pre_norm: bool = True,
        post_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(normalized_shape, eps=eps, **kwargs)
        self.pre_norm = pre_norm
        self.post_norm = post_norm

    def forward(self, input: torch.Tensor, mode: str = "direct") -> torch.Tensor:
        """
        Apply normalization based on mode.

        Args:
            input: Input tensor
            mode: "pre", "post", "both", "none", or "direct"

        Returns:
            Normalized tensor
        """
        if mode == "pre" and self.pre_norm:
            return super().forward(input)
        elif mode == "post" and self.post_norm:
            return super().forward(input)
        elif mode == "both" and (self.pre_norm or self.post_norm):
            return super().forward(input)
        elif mode == "direct":
            return super().forward(input)
        elif mode == "none":
            return input
        else:
            # No normalization applied for this mode/config combination
            return input


class PostRMSNorm(RMSNorm):
    """RMSNorm that only applies post-normalization."""
    
    def __init__(
        self,
        normalized_shape: Any,
        eps: float = 1e-05,
        **kwargs: Any,
    ) -> None:
        super().__init__(normalized_shape, eps=eps, pre_norm=False, post_norm=True, **kwargs)


class SandwichNorm(RMSNorm):
    """Sandwich normalization that applies both pre and post normalization."""
    
    def __init__(
        self,
        normalized_shape: Any,
        eps: float = 1e-05,
        **kwargs: Any,
    ) -> None:
        super().__init__(normalized_shape, eps=eps, pre_norm=True, post_norm=True, **kwargs)
