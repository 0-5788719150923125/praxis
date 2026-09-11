"""Decode-length bucketing: round every decode forward up to a fixed ladder.

WHY THIS EXISTS. On a byte-latent model a turn walks a new sequence length on
every step, and each crossing of a patch boundary is also a new trunk length.
That is fine for eager, and ruinous for anything that specializes on shape:
``praxis.memory.neural_memory.decode_compiled`` compiles with static shapes, so
an unbucketed turn woke Inductor's compile pool dozens of times and drove child
memory from ~600 MB to ~2.8 GB (see ``ModelBackend.__init__`` for the run that
died of it). Bucketing is the precondition that comment names.

WHY IT IS NEARLY FREE. Measured on abstractinator-r (RTX 5060 Ti, fp32, eager):
one forward costs 32 ms at 8 bytes, 42 ms at 1024 and 60 ms at 4088. Roughly
60% of a decode forward is length-INDEPENDENT dispatch, so a few hundred extra
positions cost a few percent while the shape set collapses by an order of
magnitude - a 64-byte turn from a 1000-byte prompt walks 41 distinct lengths and
lands on 2 rungs. "Nearly" is why the ladder has a constant stride rather than
doubling, and why the backend only turns this on when something downstream is
actually shape-sensitive.

WHY IT IS SAFE. Every stage of this stack is causal under append - the space
and static patchers are prefix-monotone, the local conv encoder/decoder are
causal, and ``decoder_patch_ids_from_lengths`` drops patch 0 so no byte reads an
open patch - so positions appended AFTER the last real byte cannot reach it.
What padding does move is the handful of mechanisms that read the live sequence
LENGTH (Kaleidoscope resamples its ratio-coordinate mirrors to ``[T, T]``; the
memory's chunk grid shifts). Measured on the live checkpoint across 23
(length, bucket) pairs, the next-byte logits move by at most 9.4e-5 relative
with zero argmax flips and identical top-5, and greedy turns are byte-identical
at strides 64 and 256 across nine prompt/length combinations. This is the same
padding-invariance ``verify_prefixes_batched`` already relies on to verify a
batch of ragged prefixes in one forward.

Scoped rather than global: only a generation that opens :func:`decode_buckets`
pads, so training and validation are untouched by construction.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Optional, Sequence, Tuple

import torch

__all__ = [
    "DECODE_BUCKETS",
    "decode_buckets",
    "active_buckets",
    "bucket_length",
    "pad_for_decode",
]

# The ladder: two short rungs, then a constant stride. A GEOMETRIC ladder is
# the obvious choice and it is the wrong one - its overshoot is proportional to
# the length, so a 1070-byte turn pads to 2048 and pays ~18% more per forward
# for nothing. A constant stride caps the overshoot in absolute terms, which is
# what matters when the forward is nearly flat in length: at stride 256 the
# worst case is ~256 extra positions, about 3% of a forward at the lengths a
# chat turn actually occupies, and a whole 4096-byte context still collapses to
# 18 shapes rather than 512 (one per patch boundary it crosses).
#
# Every rung is a multiple of any patch size this repo uses, so the trunk
# length lands on the same small set the byte length does.
DECODE_BUCKET_STRIDE: int = 256
DECODE_BUCKETS: Tuple[int, ...] = (64, 128) + tuple(
    range(DECODE_BUCKET_STRIDE, 8192 + 1, DECODE_BUCKET_STRIDE)
)

_state = threading.local()


def _rungs() -> Optional[Sequence[int]]:
    return getattr(_state, "rungs", None)


def active_buckets() -> Optional[Sequence[int]]:
    """The ladder in force on this thread, or None when bucketing is off."""
    return _rungs()


@contextmanager
def decode_buckets(
    enabled: bool = True,
    rungs: Sequence[int] = DECODE_BUCKETS,
    cap: Optional[int] = None,
):
    """Bucket decode-forward lengths for the duration of the block.

    ``cap`` is the model's positional capacity; rungs above it are dropped
    rather than clamped, because padding past what the model can represent is
    not a shape it should ever compile for.
    """
    if not enabled:
        yield
        return
    ladder = tuple(r for r in sorted(rungs) if cap is None or r <= cap)
    previous = _rungs()
    _state.rungs = ladder or None
    try:
        yield
    finally:
        _state.rungs = previous


def bucket_length(length: int, rungs: Optional[Sequence[int]] = None) -> int:
    """The padded length ``length`` rounds up to, or ``length`` unchanged.

    Past the top rung the ladder stops: a sequence longer than every rung is
    left alone rather than padded to some multiple, because the only lengths
    above the top rung are ones the positional cap already bounds.
    """
    ladder = rungs if rungs is not None else _rungs()
    if not ladder:
        return int(length)
    for rung in ladder:
        if length <= rung:
            return int(rung)
    return int(length)


def pad_for_decode(
    input_ids: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """Right-pad ``input_ids`` to its bucket. Returns ``(ids, mask, true_len)``.

    ``true_len`` is the caller's original length and is what every output has
    to be trimmed back to; it equals ``input_ids.size(1)`` when nothing was
    padded, so the trim is unconditional and cheap.

    The pad value is 0. Under a pure 256-byte alphabet that is a real byte, and
    it is inert for the reason the module docstring gives - causality, not the
    value - which is also why no mask is SYNTHESIZED here. A caller that
    already passes a mask gets its pad gated off; a caller that passes none
    keeps passing none, so the unbucketed and bucketed calls differ in exactly
    one thing.
    """
    if input_ids is None or input_ids.dim() < 2:
        return (
            input_ids,
            attention_mask,
            0 if input_ids is None else int(input_ids.size(-1)),
        )
    true_len = int(input_ids.size(1))
    pad = bucket_length(true_len) - true_len
    if pad <= 0:
        return input_ids, attention_mask, true_len
    input_ids = torch.cat(
        [input_ids, input_ids.new_zeros((input_ids.size(0), pad))], dim=1
    )
    if attention_mask is not None and attention_mask.dim() == 2:
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_zeros((attention_mask.size(0), pad))],
            dim=1,
        )
    return input_ids, attention_mask, true_len
