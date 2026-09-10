"""Sandwich normalization: a norm at the pre position and again at the post."""

from typing import Any, Type

import torch
import torch.nn as nn

from praxis.normalization.layer_norm import LayerNorm
from praxis.normalization.rms_norm import RMSNorm


class SandwichNorm(RMSNorm):
    """One RMSNorm at both positions, sharing a single weight between them."""

    def __init__(
        self,
        normalized_shape: Any,
        eps: float = 1e-05,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            normalized_shape, eps=eps, pre_norm=True, post_norm=True, **kwargs
        )


class PairedNorm(nn.Module):
    """A sandwich built from two independent norms, one per position.

    ``pre_cls`` reads the residual into the sublayer and ``post_cls`` rescales
    what the sublayer writes back, so the two positions can differ in type as
    well as in weight. ``mode`` selects the position; ``"direct"`` and
    ``"both"`` use the pre norm and ``"none"`` is a no-op.
    """

    pre_cls: Type[nn.Module] = RMSNorm
    post_cls: Type[nn.Module] = RMSNorm

    def __init__(
        self,
        normalized_shape: Any,
        eps: float = 1e-05,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.pre_norm = True
        self.post_norm = True
        self.pre = self.pre_cls(normalized_shape, eps=eps, **kwargs)
        self.post = self.post_cls(normalized_shape, eps=eps, **kwargs)

    def forward(self, input: torch.Tensor, mode: str = "direct") -> torch.Tensor:
        if mode == "none":
            return input
        elif mode == "post":
            return self.post(input, mode="direct")
        else:
            return self.pre(input, mode="direct")


class UntiedSandwichNorm(PairedNorm):
    """SandwichNorm with a separate RMSNorm weight at each position."""


class HeroNorm(PairedNorm):
    """LayerNorm on the read, RMSNorm on the write.

    Centering conditions what the sublayer sees, while the residual write is
    rescaled without having its mean stripped - so the branch keeps the freedom
    to move the stream's mean, which is the one thing RMSNorm never touches.
    """

    pre_cls = LayerNorm
    post_cls = RMSNorm


class InvertedHeroNorm(PairedNorm):
    """RMSNorm on the read, LayerNorm on the write: the hero's mirror image."""

    pre_cls = RMSNorm
    post_cls = LayerNorm
