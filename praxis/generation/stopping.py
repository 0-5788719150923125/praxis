"""Stop-string halting for decode loops that own their own sampling.

Control-token formats halt on a token id, which is a single integer compare.
Text-boundary formats (``chat_format: prose``) halt on a STRING - the role name
of the next speaker on its own line - so halting needs the decoded tail rather
than the last id.

``transformers`` already implements this for its native ``generate`` loop
(``GenerationConfig.stop_strings`` + ``StopStringCriteria``), and the
``ModelBackend`` path gets it for free because it passes ``tokenizer=`` through.
The two loops that sample for themselves - byte-level speculative decoding in
``PraxisForCausalLM._speculative_generate`` and the Ray-actor
``MonoForwardBackend`` - do not, and a format whose boundaries a loop cannot see
runs to ``max_new_tokens`` on every turn. This module is what they share.
"""

import time
from typing import Optional, Sequence, Tuple

import torch
from transformers import StoppingCriteria


class DeadlineCriteria(StoppingCriteria):
    """Halt a decode once wall-clock ``deadline`` passes.

    The queued generation path runs inside the training loop, so a request's
    cost is the run's cost. A deadline checked only BETWEEN halt-and-resume
    steps does not bound that: a plain turn with no tool call is a single
    ``generate_until_halt``, so the check would fire exactly once, before any
    token. This is the same bound applied per decoding step, which is where the
    time actually goes.

    Halting here is indistinguishable from halting on length: the sequence is
    returned as normal and the partial turn extracts normally, because
    ``reply_start`` is the runtime's own offset rather than something recovered
    from a clean stop.
    """

    def __init__(self, deadline: float) -> None:
        self.deadline = float(deadline)

    def __call__(self, input_ids, scores, **kwargs):
        expired = time.time() >= self.deadline
        return torch.full(
            (input_ids.shape[0],), expired, dtype=torch.bool, device=input_ids.device
        )


def deadline_criteria(deadline: Optional[float]):
    """``StoppingCriteriaList`` carrying a deadline, or None when unbounded."""
    if deadline is None:
        return None
    from transformers import StoppingCriteriaList

    return StoppingCriteriaList([DeadlineCriteria(deadline)])


def _window(stop_strings: Sequence[str]) -> int:
    """Tokens to decode when testing for a stop-string suffix.

    Every token decodes to at least one character (specials included, since we
    decode with ``skip_special_tokens=False``), so a window as long as the
    longest stop string always contains it. The slack absorbs multi-character
    tokens straddling the start of the match.
    """
    return max((len(s) for s in stop_strings), default=0) + 4


def find_stop_cut(
    tokenizer,
    ids: Sequence[int],
    start_index: int,
    stop_strings: Sequence[str],
) -> Optional[int]:
    """Length to truncate ``ids`` to so it ends at the first completed stop
    string, or ``None`` if none completed.

    Only positions AFTER ``start_index`` are considered. That bound is what
    makes resumption work: after halting on ``\\n\\ncall\\n\\n`` the sequence
    already ends with a stop string, and re-halting on it would return zero new
    tokens forever. ``transformers`` gets the same property by evaluating its
    criteria only after a token has been appended.

    Multi-token commits (a speculative run) can complete a stop string in the
    middle of the run, so the scan walks one position at a time and returns the
    EARLIEST completion - anything the model drafted past the boundary belongs
    to a turn that is not its to write.
    """
    if not stop_strings:
        return None
    ids = list(ids)
    window = _window(stop_strings)
    for k in range(max(0, start_index) + 1, len(ids) + 1):
        lo = max(0, k - window)
        tail = tokenizer.decode(ids[lo:k], skip_special_tokens=False)
        for stop in stop_strings:
            if tail.endswith(stop):
                return k
    return None


def normalize_stop_strings(value) -> Tuple[str, ...]:
    """Coerce a ``stop_strings`` config value (None / str / sequence) to a tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def split_at_stop(text: str, stop_strings: Sequence[str]) -> str:
    """``text`` up to the first stop string, with the boundary removed.

    Used when presenting a reply: the halt leaves the boundary that ended the
    turn attached to the text, and a chat client should not see the next
    speaker's name.
    """
    cut = len(text)
    for stop in stop_strings:
        idx = text.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def trailing_stop(text: str, stop_strings: Sequence[str]) -> Optional[str]:
    """Which stop string ``text`` ends with, if any.

    Longest match wins, so a boundary that is a suffix of another (none today,
    but role names are data) classifies unambiguously.
    """
    matches = [s for s in stop_strings if s and text.endswith(s)]
    if not matches:
        return None
    return max(matches, key=len)


__all__ = [
    "DeadlineCriteria",
    "deadline_criteria",
    "find_stop_cut",
    "normalize_stop_strings",
    "split_at_stop",
    "trailing_stop",
]
