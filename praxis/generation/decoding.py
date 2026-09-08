"""Shared pieces of a transformers-compatible decoding method.

Praxis has decode loops that are not ``_sample``: MTP speculative decoding and
CALM's patch vote. Both are registered as transformers *decoding methods* and
called as

    method(model, input_ids, logits_processor=..., stopping_criteria=...,
           generation_config=..., **model_kwargs)

which means transformers has already built the full ``LogitsProcessorList``
(repetition penalty, suppressed tokens, temperature/top-k/top-p under sampling,
renormalization) and the full ``StoppingCriteriaList`` (max length, max time,
stop strings, EOS). A decoding method must USE those rather than rebuild them -
rebuilding is how the three loops drifted apart in the first place, and it is
why a knob honored on one path was silently ignored on another.

Two things ``_sample`` gets for free that a Praxis loop does not:

- ``_sample`` picks one token per step, so evaluating the criteria after each
  append is exact. Our loops COMMIT SEVERAL TOKENS AT ONCE (a speculative run,
  a K-byte CALM patch), so a boundary can complete in the middle of a commit
  and the criteria only report that it completed somewhere. :func:`first_halt`
  recovers the position.
- ``_sample`` is the definition of "how a token is chosen". :func:`pick_next`
  is that same body, so a custom loop samples identically rather than
  approximately.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)

__all__ = [
    "pick_next",
    "is_halted",
    "first_halt",
    "split_positional_criteria",
    "stream_put",
    "stream_end",
    "trunk_hooks",
]


def trunk_hooks(model) -> dict:
    """The trunk closures an encoder-owned decoding loop needs, from a model.

    An encoder that drives generation still has to reach the global
    transformer, and it deliberately holds no reference to it - the
    Abstractinator does not know the head exists, and the model owns the
    decoder. Handing over closures per call rather than storing them is what
    keeps that separation.

    - ``base_forward(ids)`` runs the trunk from TOKENS and returns an output
      exposing ``last_hidden_state`` (plus ``patch_embeds``, ``h_encoder``,
      ``patch_lengths`` and ``local_decoder_tokens`` for encoders that patch).
    - ``latent_forward(patch_embeds, positions=None)`` runs the trunk directly
      on a LATENT sequence. An encoder that autoregresses over patches needs
      it: the patch it just predicted has no bytes behind it, so
      ``base_forward`` cannot reach it.
    - ``decode_logits(embeds)`` applies the model's head.

    Returns a dict so a caller can splice in its own ``base_forward`` (the
    Mono-Forward in-process trainer runs the trunk as a chain of workers) and
    keep the rest.
    """
    from praxis.containers import LossContainer
    from praxis.modeling import PraxisModel

    return {
        "base_forward": lambda ids: PraxisModel.forward(model, input_ids=ids),
        "latent_forward": lambda pe, positions=None: model.decoder(
            pe, None, None, None, None, LossContainer(), None, positions
        )[0],
        "decode_logits": lambda embeds: model.head(embeds),
    }


# Criteria whose verdict is a pure function of the ids in the sequence, so
# asking them about a PREFIX is a meaningful question ("had we stopped here,
# would this have halted?"). These are the ones a multi-token commit can be
# walked against to find where the halt actually landed.
#
# Everything else is evaluated on the full sequence only and never truncates.
# ``MaxTimeCriteria`` is the reason this partition exists rather than a blanket
# walk: it answers "is the clock past the deadline" identically at every
# prefix, so a blanket walk would see it fire at the first position tested and
# throw away the entire commit. A deadline halt is a length halt - keep what
# was produced and stop - which is what leaving it out of the walk gives.
#
# An unrecognized criterion (a caller's own subclass) is deliberately treated
# as non-positional: halting on it is honored, but it never truncates, because
# we cannot know that evaluating it at a prefix means anything.
_POSITIONAL_CRITERIA: Tuple[type, ...] = (
    MaxLengthCriteria,
    StopStringCriteria,
    EosTokenCriteria,
)


def pick_next(
    raw_logits: torch.Tensor,
    context_ids: torch.Tensor,
    logits_processor=None,
    do_sample: bool = False,
) -> torch.LongTensor:
    """Choose one token per row, exactly as ``GenerationMixin._sample`` does.

    ``raw_logits`` is ``[B, vocab]`` (the next-token logits, already selected
    from whatever position the caller cares about) and ``context_ids`` is the
    ``[B, T]`` prefix the processors score against - context-dependent
    processors like the repetition penalty read it, so it must be the prefix
    THIS token follows, not the sequence as a whole.

    The float32 cast matches ``_sample``: a bf16 model's logits go through the
    processors and the softmax in float32 there, and sampling is sensitive
    enough to the difference that skipping it would make a "shared" sampler
    quietly disagree with the standard path.

    Temperature is NOT applied here. Under ``do_sample`` transformers has
    already put a ``TemperatureLogitsWarper`` in ``logits_processor``, so
    dividing again would square the effect - which is exactly the bug the old
    hand-rolled loops carried the risk of once they started receiving a
    prepared list.
    """
    scores = raw_logits.to(copy=True, dtype=torch.float32, device=context_ids.device)
    if logits_processor is not None:
        scores = logits_processor(context_ids, scores)
    if do_sample:
        probs = torch.nn.functional.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)
    return torch.argmax(scores, dim=-1)


def is_halted(seq: torch.Tensor, criteria: Optional[StoppingCriteriaList]) -> bool:
    """Whether every row of ``seq`` has met a stopping criterion.

    ``scores`` is passed as None: none of the criteria a decoding method
    receives read it (``ConfidenceCriteria`` does, but it is only built for an
    assistant model, which never drives one of our loops).
    """
    if not criteria:
        return False
    return bool(criteria(seq, None).all())


def split_positional_criteria(
    criteria: Optional[StoppingCriteriaList],
) -> Tuple[StoppingCriteriaList, StoppingCriteriaList]:
    """Split a prepared list into ``(positional, whole_sequence)``.

    See :data:`_POSITIONAL_CRITERIA`. Both halves are real
    ``StoppingCriteriaList`` objects so either can be called directly.
    """
    positional = StoppingCriteriaList()
    whole = StoppingCriteriaList()
    for criterion in criteria or ():
        if isinstance(criterion, _POSITIONAL_CRITERIA):
            positional.append(criterion)
        else:
            whole.append(criterion)
    return positional, whole


def first_halt(
    seq: torch.Tensor,
    criteria: Optional[StoppingCriteriaList],
    start_index: int,
) -> Optional[int]:
    """Length to truncate ``seq`` to so it ends at the EARLIEST halt, or None.

    Only positions strictly after ``start_index`` are considered, and that
    bound is what makes halt-and-resume work: after halting on a boundary the
    sequence already ends in one, so re-testing it would return the same
    position forever and the loop would never advance. ``_sample`` gets the
    same property for free by only ever evaluating a freshly appended token;
    a multi-token commit has to say where its own new content starts.

    Anything the model drafted past the boundary belongs to a turn it does not
    get to write, so the earliest completion wins rather than the last.

    Non-positional criteria (a deadline) are ignored here on purpose - they
    halt the loop but never truncate it. Callers pair this with
    :func:`is_halted` over the whole list.
    """
    positional, _ = split_positional_criteria(criteria)
    if not positional:
        return None
    total = seq.shape[-1]
    for k in range(max(0, start_index) + 1, total + 1):
        if bool(positional(seq[..., :k], None).all()):
            return k
    return None


def stream_put(streamer, ids: torch.Tensor) -> None:
    """Publish committed ids to a streamer, if there is one.

    ``ids`` is ``[B, n]`` or ``[B]``. ``BaseStreamer.put`` is documented against
    what ``_sample`` hands it - a ``[B]`` tensor of one step's tokens - so a
    multi-token commit is published column by column rather than as a block.
    Consumers that decode incrementally (every one of them) would otherwise
    have to guess which shape they were being given.
    """
    if streamer is None or ids is None:
        return
    if ids.dim() == 1:
        streamer.put(ids)
        return
    for i in range(ids.shape[-1]):
        streamer.put(ids[..., i])


def stream_end(streamer) -> None:
    """Signal end-of-generation, if there is a streamer.

    A decoding method owns this: ``generate`` publishes the PROMPT before
    dispatch and ``_sample`` calls ``end()`` on its way out, so a custom method
    that skips it leaves every consumer waiting on a turn that already
    finished.
    """
    if streamer is not None:
        streamer.end()
