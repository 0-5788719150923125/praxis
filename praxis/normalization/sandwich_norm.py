"""Sandwich normalization that applies both pre and post normalization."""

from typing import Any

from praxis.normalization.rms_norm import RMSNorm


class SandwichNorm(RMSNorm):
    """Sandwich normalization that applies both pre and post normalization."""

    def __init__(
        self,
        normalized_shape: Any,
        eps: float = 1e-05,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            normalized_shape, eps=eps, pre_norm=True, post_norm=True, **kwargs
        )
