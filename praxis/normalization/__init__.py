"""Normalization module with various normalization implementations."""

from praxis import registry
from praxis.normalization.base import BaseNorm, NoNorm
from praxis.normalization.layer_norm import LayerNorm
from praxis.normalization.rms_norm import PostRMSNorm, RMSNorm
from praxis.normalization.sandwich_norm import (
    HeroNorm,
    InvertedHeroNorm,
    PairedNorm,
    SandwichNorm,
    TiedSandwichNorm,
)
from praxis.registry import Entry

registry.declare(
    "normalization",
    title="Normalization layers",
    doc=(
        (
            "LayerNorm/RMSNorm variants, including SandwichNorm (required for stable "
            "recurrent-depth bias). Each transformer sublayer (attention, feedforward) has "
            "two norm positions: pre, on its input, and post, on its output before it "
            "joins the residual. An entry decides which positions are filled and with "
            "what. There is no final norm before the head."
        )
    ),
    entries={
        "none": Entry(
            NoNorm,
            "No normalization at either position.",
        ),
        "layer_norm": Entry(
            LayerNorm,
            "LayerNorm at the pre position only.",
        ),
        "rms_norm": Entry(
            RMSNorm,
            "RMSNorm at the pre position only.",
        ),
        "post_rms_norm": Entry(
            PostRMSNorm,
            "RMSNorm at the post position only.",
        ),
        "sandwich": Entry(
            SandwichNorm,
            "RMSNorm at both positions, with a separate weight for each.",
        ),
        "sandwich_tied": Entry(
            TiedSandwichNorm,
            "RMSNorm at both positions, sharing a single weight between them.",
        ),
        "hero": Entry(
            HeroNorm,
            (
                "Both positions, mixed types: LayerNorm at pre, RMSNorm at post. "
                "Centering conditions what the sublayer reads, while the write back to "
                "the residual is rescaled without having its mean stripped, so the "
                "branch keeps the freedom to move the stream's mean."
            ),
        ),
        "hero_inverted": Entry(
            InvertedHeroNorm,
            (
                "Both positions, hero's mirror image: RMSNorm at pre, LayerNorm at "
                "post."
            ),
        ),
    },
)

__all__ = [
    "BaseNorm",
    "NoNorm",
    "LayerNorm",
    "RMSNorm",
    "PostRMSNorm",
    "PairedNorm",
    "SandwichNorm",
    "TiedSandwichNorm",
    "HeroNorm",
    "InvertedHeroNorm",
]
