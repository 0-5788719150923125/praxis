from functools import partial
from typing import Any, Dict

from praxis.normalization.base import LayerNorm, NoNorm, RMSNorm

# Base normalization types with partial for pre_norm and post_norm flags
NORMALIZATION_REGISTRY: Dict[str, Any] = {
    "none": NoNorm,
    "layer_norm": LayerNorm,
    "rms_norm": RMSNorm,
    "post_rms_norm": partial(RMSNorm, pre_norm=False, post_norm=True),
    "sandwich": partial(LayerNorm, pre_norm=True, post_norm=True),
}
