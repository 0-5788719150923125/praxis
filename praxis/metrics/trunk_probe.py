"""Does the trunk's output carry anything the decoder reads?

Give every row the trunk output computed for ANOTHER row and measure what that
costs. A magnitude reading cannot answer this: the trunk's contribution can be a
large vector and still be the same vector for every input, in which case swapping
it between rows is free. The swap is the counterfactual the magnitude hides -
bits lost when the trunk's input-dependence is destroyed while its scale, its
positional shape and every other path are untouched.

Zero means the trunk is a constant as far as the decoder is concerned, whatever
its norm; a positive value is the part of the prediction that came through it.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _nll_bits(model: nn.Module, input_ids: Tensor, aligned: bool) -> Optional[Tensor]:
    labels = input_ids if aligned else input_ids[:, 1:].contiguous()
    logits = getattr(model(input_ids=input_ids, labels=labels), "logits", None)
    if logits is None or logits.dim() != 3:
        return None
    if not aligned:
        logits = logits[..., :-1, :]
    if logits.shape[:-1] != labels.shape:
        return None
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        reduction="none",
    ).view(labels.shape).mean() / math.log(2.0)


@torch.no_grad()
def trunk_swap_cost(
    model: nn.Module, input_ids: Tensor, aligned: bool = False
) -> Optional[Tensor]:
    """Bits per token lost when each row is decoded on another row's trunk output.

    Runs the uncompiled module, so the probe never spends the compiled model's
    recompile budget. None when the model has no separable trunk (no encoder to
    decode its output), when the batch has fewer than two rows to swap, or when
    the model emits no per-token logits.
    """
    model = getattr(model, "_orig_mod", model)
    decoder = getattr(model, "decoder", None)
    if getattr(model, "encoder", None) is None or decoder is None:
        return None
    if input_ids.size(0) < 2:
        return None

    base = _nll_bits(model, input_ids, aligned)
    if base is None:
        return None

    forward = decoder.forward

    def rolled(*args, **kwargs):
        out = forward(*args, **kwargs)
        if isinstance(out, tuple) and torch.is_tensor(out[0]) and out[0].dim() == 3:
            return (out[0].roll(1, dims=0),) + tuple(out[1:])
        return out

    decoder.forward = rolled
    try:
        swapped = _nll_bits(model, input_ids, aligned)
    finally:
        decoder.forward = forward
    if swapped is None:
        return None
    return swapped - base
