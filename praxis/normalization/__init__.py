from functools import partial
from typing import Any, Dict

from praxis.normalization.base import LayerNorm, NoNorm, RMSNorm

# Base normalization types with partial for pre_norm and post_norm flags
NORMALIZATION_REGISTRY: Dict[str, Any] = {
    # Basic normalization types - default to pre-norm behavior
    "none": NoNorm,
    "layer_norm": LayerNorm,
    "rms_norm": RMSNorm,
    # Explicit post-norm profiles (original transformer style)
    "post_rms_norm": partial(RMSNorm, pre_norm=False, post_norm=True),
}
