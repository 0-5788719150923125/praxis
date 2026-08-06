"""Generation request/result data structures."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GenerationRequest:
    """Represents a text generation request."""

    id: str
    prompt: str
    kwargs: Dict[str, Any]
    result: Optional[str] = None


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
