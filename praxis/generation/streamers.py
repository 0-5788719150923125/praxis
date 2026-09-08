"""Incremental publication of a model's reply, as a transformers streamer.

``generate(streamer=...)`` is how transformers publishes tokens as they are
produced, and every Praxis decode path now honors it - the standard loop
natively, the speculative and CALM decoding methods through
:func:`praxis.generation.decoding.stream_put`. What a *consumer* wants, though,
is not tokens: it is the model's reply as readable text, with the runtime's own
plumbing (turn boundaries, spliced tool exchanges) left out. That is this.

THE ONE RULE HERE: the visible text is computed by the SAME
``extract_assistant_reply`` that produces the final answer, run over the tokens
seen so far. There is no second, streaming-flavored copy of the extraction
logic - that duplication is what would drift, and the whole point of publishing
incrementally is that the last delta plus everything before it equals the
answer the caller would have gotten by waiting.

What incremental publication *does* add is a hold-back. A boundary the extractor
would cut at arrives one token at a time, so a naive flush emits the first half
of ``\\n\\nuser\\n\\n`` before anyone knows it was a boundary, and published
text cannot be retracted. :class:`ReplyStreamer` therefore keeps the last
``holdback`` characters private until either more tokens prove them ordinary or
the turn ends.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import torch
from transformers.generation.streamers import BaseStreamer

from praxis.tokenizers.chat_templates import chat_format_of

__all__ = ["ReplyStreamer"]


class ReplyStreamer(BaseStreamer):
    """Publishes the model's reply as text deltas while it is being decoded.

    Driven by :class:`praxis.generation.generator.Generator`, which owns the
    halt-and-resume turn this spans. Two departures from the plain
    ``BaseStreamer`` contract follow from that, and both are deliberate:

    - ``end()`` is a STEP boundary, not the end of the turn. A Praxis turn is
      several ``generate`` calls - it halts at each tool boundary and resumes -
      and every one of them calls ``end()`` on its way out. :meth:`finish` is
      what ends the turn.
    - each of those calls also re-publishes the whole sequence so far as its
      prompt (``generate`` does ``streamer.put(input_ids)`` before dispatch),
      so the driver calls :meth:`begin_step` first and that publish is skipped.

    ``on_text`` receives each delta - never the whole text - so a consumer
    appends rather than replaces. ``on_reset`` is the other half of that
    contract: it fires when everything published so far stops being part of the
    answer, which happens exactly once per tool call. The runtime's turn anchor
    moves past each spliced tool result (``GenerationResult.reply_start``), so
    the final reply is only what the model wrote AFTER the last result - a
    consumer that appended the pre-call chatter has to drop it, and this is how
    it is told to. A consumer with nowhere to put a reset can omit it and treat
    the final result as authoritative instead.
    """

    # Tool plumbing the reply extractor strips wholesale. Held back like a
    # boundary so a half-written marker is never published: under
    # ``tool_style="tokens"`` the runtime halts on ``[TOOL_CALL]`` only AFTER
    # the model has written it, so by then it would already be on the wire.
    _TOOL_MARKERS = ("[TOOL_CALL]", "[/TOOL_CALL]", "[TOOL_RESULT]", "[/TOOL_RESULT]")

    def __init__(
        self,
        tokenizer: Any,
        on_text: Callable[[str], None],
        on_reset: Optional[Callable[[], None]] = None,
        holdback: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.on_text = on_text
        self.on_reset = on_reset
        self._ids: List[int] = []
        self._published = ""
        self._skip_next_put = False
        self._muted = False
        self.holdback = (
            self._default_holdback(tokenizer) if holdback is None else int(holdback)
        )

    # ------------------------------------------------------------------
    # driver hooks
    # ------------------------------------------------------------------

    def begin_step(self) -> None:
        """Announce a new ``generate`` call, whose prompt publish is not ours.

        Every step re-publishes the entire sequence built so far - the original
        prompt, everything the model has written, and any tool result the
        runtime spliced. Only what comes after is new.
        """
        self._skip_next_put = True

    def mute(self, muted: bool = True) -> None:
        """Stop (or resume) absorbing tokens.

        The driver mutes for the duration of a tool call. What the model writes
        there is a JSON body the reply never contains, and the extractor can
        only recognise it once the block CLOSES - so a streamer left listening
        would publish a half-written call before anything could decide to strip
        it. Published text cannot be retracted; not publishing it can.
        """
        self._muted = bool(muted)

    def restart(self) -> None:
        """Everything published so far has stopped being part of the answer.

        Fired once per tool call, when the runtime splices the result and moves
        the turn anchor past it: from that point the reply is only what the
        model writes next, so the buffer is cleared and the consumer is told to
        drop what it has.
        """
        self._ids = []
        self._published = ""
        self._muted = False
        if self.on_reset is not None:
            self.on_reset()

    def finish(self) -> None:
        """End the turn: publish whatever the hold-back was still keeping."""
        self._flush(final=True)

    @property
    def text(self) -> str:
        """Everything published so far."""
        return self._published

    # ------------------------------------------------------------------
    # BaseStreamer
    # ------------------------------------------------------------------

    def put(self, value: torch.Tensor) -> None:
        if self._skip_next_put:
            self._skip_next_put = False
            return
        if self._muted or value is None:
            return
        # Batch axis: a decode driven by the Generator is always one sequence,
        # and anything wider is a caller this streamer cannot describe - so read
        # the first row rather than interleaving several turns into one buffer.
        row = value if value.dim() == 1 else value[0]
        self._ids.extend(int(i) for i in row.reshape(-1).tolist())
        self._flush()

    def end(self) -> None:
        """One ``generate`` call finished - NOT the turn. See the class docstring."""
        self._flush()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _default_holdback(self, tokenizer) -> int:
        """Longest string the reply extractor might cut at.

        Holding back that many characters guarantees a boundary is never half
        published: any cut the extractor will make is fully visible in the
        buffer before the characters preceding it are released.
        """
        try:
            fmt = chat_format_of(tokenizer)
        except Exception:
            return 16
        candidates = [fmt.boundary(role) for role in fmt.roles]
        candidates.extend(self._TOOL_MARKERS)
        for name in ("eos_token", "sep_token", "bos_token"):
            token = getattr(tokenizer, name, None)
            if isinstance(token, str) and token:
                candidates.append(token)
        try:
            for token_id in fmt.stop_token_ids(tokenizer):
                candidates.append(
                    tokenizer.decode([token_id], skip_special_tokens=False)
                )
        except Exception:
            pass
        return max((len(c) for c in candidates if c), default=0)

    def _decode(self) -> str:
        """The model's text so far, without a severed multi-byte tail.

        A byte-level tokenizer emits one byte per token, so a decode taken
        mid-character comes back with U+FFFD - which would be published and
        then contradicted by the real character a token later. Tokenizers that
        know their own layout trim the incomplete tail; the rest have nothing
        to fix.
        """
        ids = self._ids
        strip = getattr(self.tokenizer, "strip_incomplete_tail", None)
        if strip is not None:
            try:
                ids = strip(list(ids))
            except Exception:
                ids = self._ids
        if not ids:
            return ""
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def _visible(self) -> str:
        """The reply the caller would get if the turn ended right now.

        Delegated to the real extractor rather than reimplemented, so a
        streamed turn and a waited-for turn cannot disagree. The empty-turn
        placeholder is a presentation choice for a finished turn, so it is
        never streamed.
        """
        from praxis.generation.reply import (
            EMPTY_REPLY_PLACEHOLDER,
            extract_assistant_reply,
        )
        from praxis.generation.request import GenerationResult

        raw = self._decode()
        if not raw:
            return ""
        try:
            # reply_start = 0: this buffer holds only what the MODEL wrote, so
            # the turn starts at its first character. Handing that in is what
            # stops the extractor from scanning for a boundary the model may
            # have written itself.
            reply = extract_assistant_reply(GenerationResult(raw, 0), self.tokenizer)
        except Exception:
            return ""
        return "" if reply == EMPTY_REPLY_PLACEHOLDER else reply

    def _flush(self, final: bool = False) -> None:
        visible = self._visible()
        if not final and self.holdback:
            visible = visible[: max(0, len(visible) - self.holdback)]
        if not visible.startswith(self._published):
            # The extractor revised what it had already shown - a boundary
            # completed inside text we released, which the hold-back is sized
            # to prevent. Publish nothing further rather than contradict what
            # the consumer already has; `finish` and the caller's final answer
            # remain authoritative.
            return
        delta = visible[len(self._published) :]
        if not delta:
            return
        self._published = visible
        self.on_text(delta)
