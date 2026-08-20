"""Generation request/result data structures."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GenerationRequest:
    """Represents a text generation request.

    ``deadline`` is a wall-clock time (``time.time()`` scale) past which the
    request is not worth serving. It exists because the wait on the caller's
    side is advisory: ``generate_from_messages`` gives up after its timeout,
    but the queued request lives on and is served inline in the training loop
    by ``GenerationQueueCallback``, which has the model's turn. Without a
    deadline ON THE REQUEST, abandoning the wait only stops us listening - the
    training loop still pays for the whole generation. Measured on
    ``abstractinator-r``: one 512-byte Discord turn stalled the loop for 208
    seconds, roughly 148 of them after the client had already timed out and
    thrown the (eventual) reply away.

    ``None`` means no deadline, which is what callers that poll forever
    (the ``/input`` route) need.
    """

    id: str
    prompt: str
    kwargs: Dict[str, Any]
    result: Optional[str] = None
    deadline: Optional[float] = None


class GenerationResult(str):
    """The decoded sequence, tagged with where the model's turn starts.

    Subclasses ``str`` so every consumer that just wants text - the ``/input``
    route, the terminal's rolling contexts, the integrations - keeps working
    unchanged. Callers that have to separate the model's reply from everything
    the runtime wrote read ``reply_start`` instead of scanning backwards for a
    turn boundary.

    That scan is why this exists. Under every chat format the model can WRITE
    the boundary that opens an assistant turn - ``prose`` excludes it from the
    stop strings on purpose, and ``default`` does not suppress ``[BOS]`` - so a
    backward search for "where does the assistant's turn begin" can land on the
    model's own text and silently discard everything before it. The runtime
    knows the answer exactly and does not have to guess: it wrote the prompt,
    and it wrote every tool result it spliced.

    ``reply_start`` is a character index into this string, or ``None`` when the
    producer could not establish one (a tokenizer whose piece-wise decode does
    not concatenate, say), which tells the reader to fall back to scanning.
    """

    reply_start: Optional[int]

    def __new__(cls, text: str, reply_start: Optional[int] = None):
        obj = super().__new__(cls, text)
        obj.reply_start = None if reply_start is None else int(reply_start)
        return obj
