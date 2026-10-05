"""How much a document's later tokens gain from the context before them.

Olsson et al. (2022, "In-context Learning and Induction Heads") score in-context
learning as the loss late in the context minus the loss early in it: a model that
uses its context predicts a document's 500th token better than its 50th. This
reads the same comparison off the validation forward that already runs, so it
costs no extra forward and covers every validation row.

Positions count within a document, from ``block_ids``: a packed row holds several,
and a token early in its own document has little context whatever its place in
the row. Early is positions ``[L/32, L/16)`` and late is ``[L/2, L)`` of the row
length ``L``, scored within the same document, so the document's own difficulty
cancels. Only a document that reaches ``L/2`` inside the row contributes, and a
row holds at most one. The opening of a document also differs in kind from its
middle, so the reading is not zero for a model that ignores its context - compare
runs against each other, not against zero.

Copy gain (praxis/metrics/copy_probe.py) asks one narrow question, whether the
model retrieves a passage verbatim; this asks the broad one it serves, whether
loss keeps falling as context grows.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def context_gain(
    logits: Optional[Tensor], input_ids: Tensor, block_ids: Optional[Tensor] = None
) -> Optional[Tuple[Tensor, int]]:
    """Bits per token saved late in a document against early in it, averaged
    over the rows that hold a long enough document, and how many rows that was.

    ``logits`` are a forward's unaligned outputs ``[B, T, V]`` (position t
    predicts token t+1) for ``input_ids`` ``[B, T]``. None when the shapes do not
    line up or no document reaches the late band.
    """
    batch, length = input_ids.shape
    if logits is None or logits.dim() != 3 or length < 32:
        return None
    if tuple(logits.shape[:2]) != (batch, length):
        return None
    nll = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).view(batch, length - 1) / math.log(2.0)

    if block_ids is None or block_ids.shape != input_ids.shape:
        block_ids = torch.ones_like(input_ids)
    index = torch.arange(length, device=input_ids.device).expand(batch, length)
    starts = torch.ones_like(input_ids, dtype=torch.bool)
    starts[:, 1:] = block_ids[:, 1:] != block_ids[:, :-1]
    position = index - torch.cummax(torch.where(starts, index, 0), dim=1).values

    # Each TARGET is scored by its own place in its own document.
    position, document = position[:, 1:], block_ids[:, 1:]
    late = position >= length // 2
    rows = late.any(dim=1)
    if not bool(rows.any()):
        return None
    # The one document per row that reaches the late band.
    owner = document.gather(1, late.float().argmax(dim=1, keepdim=True))
    own = document == owner
    early = own & (position >= length // 32) & (position < length // 16)
    late = own & late
    early_mean = (nll * early).sum(1) / early.sum(1).clamp_min(1)
    late_mean = (nll * late).sum(1) / late.sum(1).clamp_min(1)
    return (early_mean - late_mean)[rows].mean(), int(rows.sum())
