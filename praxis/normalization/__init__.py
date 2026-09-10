"""Normalization module with various normalization implementations."""

from typing import Any, Dict

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

# Base normalization types
NORMALIZATION_REGISTRY: Dict[str, Any] = {
    "none": NoNorm,
    "layer_norm": LayerNorm,
    "rms_norm": RMSNorm,
    "post_rms_norm": PostRMSNorm,
    "sandwich": SandwichNorm,
    "sandwich_tied": TiedSandwichNorm,
    "hero": HeroNorm,
    "hero_inverted": InvertedHeroNorm,
}

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
    "NORMALIZATION_REGISTRY",
]
