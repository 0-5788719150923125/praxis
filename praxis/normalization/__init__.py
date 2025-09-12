from typing import Any, Dict

from praxis.normalization.base import (
    LayerNorm,
    NoNorm,
    PostRMSNorm,
    RMSNorm,
    SandwichNorm,
)

# Base normalization types
NORMALIZATION_REGISTRY: Dict[str, Any] = {
    "none": NoNorm,
    "layer_norm": LayerNorm,
    "rms_norm": RMSNorm,
    "post_rms_norm": PostRMSNorm,
    "sandwich": SandwichNorm,
}
