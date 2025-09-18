"""Normalization module with various normalization implementations."""

from typing import Any, Dict

from praxis.normalization.base import BaseNorm, NoNorm
from praxis.normalization.layer_norm import LayerNorm
from praxis.normalization.rms_norm import PostRMSNorm, RMSNorm
from praxis.normalization.sandwich_norm import SandwichNorm

# Base normalization types
NORMALIZATION_REGISTRY: Dict[str, Any] = {
    "none": NoNorm,
    "layer_norm": LayerNorm,
    "rms_norm": RMSNorm,
    "post_rms_norm": PostRMSNorm,
    "sandwich": SandwichNorm,
}

__all__ = [
    "BaseNorm",
    "NoNorm",
    "LayerNorm",
    "RMSNorm",
    "PostRMSNorm",
    "SandwichNorm",
    "NORMALIZATION_REGISTRY",
]