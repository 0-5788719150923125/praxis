"""Dtype-safe additive attention masks.

An additive mask has to be built in the dtype of the scores it will be added
to. Building it in fp32 (the old ``torch.full(..., -1e9)``, which inherits
torch's default dtype) silently promotes bf16 scores back to fp32, and then the
weights no longer match the values they multiply. The magnitude has to follow
the dtype too: -1e9 is finite in fp32/bf16 but overflows fp16, whose largest
finite value is 65504.
"""

import torch
from torch import Tensor

# The value fp32 masks have always used. Kept as the ceiling so the common
# path's numbers do not move.
NEG_MASK = -1e9


def mask_fill_value(dtype: torch.dtype) -> float:
    """Largest-magnitude negative fill this dtype can hold, capped at -1e9."""
    return max(NEG_MASK, torch.finfo(dtype).min)


def additive_mask(keep: Tensor, dtype: torch.dtype) -> Tensor:
    """Turn a boolean keep-mask into an additive mask in ``dtype``.

    True (attend here) maps to 0, False to the dtype's fill value.
    """
    return torch.where(
        keep,
        torch.zeros((), dtype=dtype, device=keep.device),
        torch.full((), mask_fill_value(dtype), dtype=dtype, device=keep.device),
    )
