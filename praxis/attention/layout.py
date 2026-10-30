"""Attention layouts for trunk sequences that are not one causal stream.

The coarse-to-fine encoder (praxis/encoders/byte_latent/coarse_to_fine.py) hands
the trunk a CONTEXT stream of patch latents followed by one block per (chunk,
level) of codes to predict. Who may read whom, and where each position sits,
then depend on that structure rather than on sequence order, so the encoder
publishes a layout for the duration of a forward and ``SelfAttention`` reads it
in place of its causal, window and document masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class BlockLayout:
    """A causal context stream plus blocks that see earlier context and themselves.

    Every field is per position of the trunk sequence:

    - ``positions`` ``[T]`` float: where the position sits, in patches. Rotary
      encodings rotate by it instead of by sequence index.
    - ``group`` ``[T]`` long: -1 for the context stream, otherwise the block id.
    - ``chunk`` ``[T]`` long: the chunk a context patch belongs to, or the chunk
      a block predicts.
    - ``doc`` ``[B, T]`` long: document id. Context queries read only their own
      document; a block reads context from its chunk's first document.

    A context query reads context keys at or before it. A block query reads
    context keys from earlier chunks and keys of its own block, nothing else.
    """

    positions: Tensor
    group: Tensor
    chunk: Tensor
    doc: Tensor

    def __post_init__(self) -> None:
        # Built once and shared by every layer: a layout lives for one forward.
        self._dense: dict = {}

    def __len__(self) -> int:
        return int(self.group.numel())

    def _allows(self, b, q, kv):
        """Whether query ``q`` may read real key ``kv`` (index tensors)."""
        group, chunk, doc = self.group, self.chunk, self.doc
        q_context = group[q] < 0
        kv_context = group[kv] < 0
        same_doc = doc[b, q] == doc[b, kv]
        from_context = kv_context & same_doc
        context_query = q_context & from_context & (kv <= q)
        block_query = ~q_context & (
            (from_context & (chunk[kv] < chunk[q])) | (group[kv] == group[q])
        )
        return context_query | block_query

    def dense(self, batch_size: int) -> Tensor:
        """``[B, 1, T, T+1]`` boolean mask over the ghost-shifted key axis
        (key 0 is the ghost, always readable)."""
        if batch_size not in self._dense:
            T = len(self)
            device = self.group.device
            b = torch.arange(batch_size, device=device).view(-1, 1, 1)
            q = torch.arange(T, device=device).view(1, -1, 1)
            kv = torch.arange(T, device=device).view(1, 1, -1)
            allowed = self._allows(b, q, kv)
            ghost = torch.ones(batch_size, T, 1, dtype=torch.bool, device=device)
            self._dense[batch_size] = torch.cat([ghost, allowed], dim=-1).unsqueeze(1)
        return self._dense[batch_size]
