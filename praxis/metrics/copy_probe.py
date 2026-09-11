"""In-context copying: how much cheaper a passage is the second time.

Repeat a passage once and compare the model's loss on the two copies. Only
context can make the repeat cheaper, and a repeat far past the local receptive
field can only be copied through the model's long-range path, so a gain near
zero means the model predicts the repeat as if it had never seen it. A model
that uses its context copies almost for free, and the gain approaches the
first copy's own cost.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Tokens per copy. A byte-level model patches 8 bytes at a time and its local
# path sees a few dozen, so a copy this long puts its source out of local reach.
COPY_PROBE_LEN: int = 256


@torch.no_grad()
def copy_gain(
    model: nn.Module,
    input_ids: Tensor,
    aligned: bool = False,
    length: int = COPY_PROBE_LEN,
) -> Optional[Tensor]:
    """Bits per token saved on the second copy of each row's first ``length``
    tokens (at most half the row).

    ``aligned`` follows the trainer's convention: an aligned model emits one
    logit per input position, otherwise logits are shifted against the input.
    The first token of each copy is excluded - nothing in the context says
    where a copy starts. Runs the uncompiled module, so the probe's shape never
    spends the compiled model's recompile budget. None when the rows are too
    short, the model emits no per-token logits, or its encoder owns the loss
    (CALM's logits reconstruct the input rather than predict it).
    """
    model = getattr(model, "_orig_mod", model)
    if getattr(getattr(model, "encoder", None), "handles_loss", False):
        return None
    length = min(int(length), input_ids.size(1) // 2)
    if length < 4:
        return None
    passage = input_ids[:, :length]
    seq = torch.cat([passage, passage], dim=1)
    labels = seq if aligned else seq[:, 1:].contiguous()
    logits = getattr(model(input_ids=seq, labels=labels), "logits", None)
    if logits is None or logits.dim() != 3:
        return None
    if not aligned:
        logits = logits[..., :-1, :]
    if logits.shape[:-1] != labels.shape:
        return None
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        reduction="none",
    ).view(labels.shape) / math.log(2.0)
    token = torch.arange(labels.size(1), device=nll.device) + (0 if aligned else 1)
    first = (token >= 1) & (token < length)
    second = (token >= length + 1) & (token < 2 * length)
    return nll[:, first].mean() - nll[:, second].mean()
