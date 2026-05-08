"""Utilities for tool call tag formatting and parsing.

Tool calls are marked with atomic special tokens. The string forms below
are what the tokenizer recognizes as single-token units, so a full
``[TOOL_CALL]`` in text encodes to exactly one id and cannot be
interrupted mid-tag by an intervening sample.
"""

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Atomic special-token string forms. The tokenizer maps each of these
# to a single vocabulary id; see ``PraxisToolTokensMixin`` in
# ``praxis/tokenizers/base.py`` for the canonical list.
TOOL_CALL_OPEN = "[TOOL_CALL]"
TOOL_CALL_CLOSE = "[/TOOL_CALL]"
TOOL_RESULT_OPEN = "[TOOL_RESULT]"
TOOL_RESULT_CLOSE = "[/TOOL_RESULT]"


def format_tool_input(
    tool_name: str, arguments: Dict[str, Any], indent: Optional[int] = None
) -> str:
    """Format a tool call as ``[TOOL_CALL]JSON[/TOOL_CALL]``.

    The bracketed strings are atomic special tokens once tokenized.
    JSON is minified by default to keep tool-call bodies short in
    training data - callers wanting pretty-printed output can pass
    ``indent=2``.

    Example:
        >>> format_tool_input("calc", {"values": [1, 2], "op": "add"})
        '[TOOL_CALL]\\n{"name": "calc", "arguments": {"values": [1, 2], "op": "add"}}\\n[/TOOL_CALL]'
    """
    tool_call_data = {"name": tool_name, "arguments": arguments}
    json_str = json.dumps(tool_call_data, indent=indent)
    return f"{TOOL_CALL_OPEN}\n{json_str}\n{TOOL_CALL_CLOSE}"


def format_tool_output(result: Any) -> str:
    """Format a tool result as ``[TOOL_RESULT]\\nresult\\n[/TOOL_RESULT]``.

    The newlines around the body keep the result markers on their own
    lines so adjacent special tokens never butt up against arbitrary
    text.

    Example:
        >>> format_tool_output(42)
        '[TOOL_RESULT]\\n42\\n[/TOOL_RESULT]'
    """
    return f"{TOOL_RESULT_OPEN}\n{result}\n{TOOL_RESULT_CLOSE}"


def format_tool_call_with_result(
    tool_name: str, arguments: Dict[str, Any], result: Any, indent: Optional[int] = None
) -> str:
    """Format a complete tool call with both input and output.

    A newline always separates the call's ``[/TOOL_CALL]`` from the
    result's ``[TOOL_RESULT]`` so adjacent special tokens stay on
    different lines.

    Example:
        >>> format_tool_call_with_result("calc", {"values": [1, 2], "op": "add"}, 3)
        '[TOOL_CALL]\\n{"name": "calc", "arguments": {"values": [1, 2], "op": "add"}}\\n[/TOOL_CALL]\\n[TOOL_RESULT]\\n3\\n[/TOOL_RESULT]'
    """
    return (
        f"{format_tool_input(tool_name, arguments, indent)}\n"
        f"{format_tool_output(result)}"
    )


def get_tool_input_pattern() -> str:
    """Regex for matching a complete tool call in decoded text.

    Used by post-hoc text parsing. Generation-time detection should
    prefer the token-ID helpers below.
    """
    open_esc = re.escape(TOOL_CALL_OPEN)
    close_esc = re.escape(TOOL_CALL_CLOSE)
    return rf"{open_esc}\s*({{.*?}})\s*{close_esc}"


def get_tool_output_pattern() -> str:
    """Regex for matching a complete tool result in decoded text."""
    open_esc = re.escape(TOOL_RESULT_OPEN)
    close_esc = re.escape(TOOL_RESULT_CLOSE)
    return rf"{open_esc}(.*?){close_esc}"


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse the LAST complete tool call from decoded text.

    Example:
        >>> text = '[TOOL_CALL]\\n{"name": "calc", "arguments": {"values": [1, 2], "op": "add"}}\\n[/TOOL_CALL]'
        >>> parse_tool_call(text)
        {'name': 'calc', 'arguments': {'values': [1, 2], 'op': 'add'}}
    """
    matches = re.findall(get_tool_input_pattern(), text, re.DOTALL)
    for match in reversed(matches):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    return None


def get_unprocessed_tool_call(text: str) -> Optional[Tuple[Dict[str, Any], int]]:
    """Find the first tool call in decoded text that has no result yet.

    Returns ``(tool_data, end_position)`` where ``end_position`` is the
    character index just past the ``[/TOOL_CALL]`` match.
    """
    matches = list(re.finditer(get_tool_input_pattern(), text, re.DOTALL))
    if not matches:
        return None
    result_open_esc = re.escape(TOOL_RESULT_OPEN)
    result_prefix = re.compile(rf"^\s*{result_open_esc}")
    for match in matches:
        if result_prefix.match(text[match.end() :]):
            continue
        try:
            return (json.loads(match.group(1)), match.end())
        except json.JSONDecodeError:
            continue
    return None


def has_complete_tool_call(text: str) -> bool:
    """True iff the decoded text contains at least one complete tool call."""
    return bool(re.search(get_tool_input_pattern(), text, re.DOTALL))


def has_tool_output(text: str) -> bool:
    """True iff the decoded text contains at least one tool result."""
    return bool(re.search(get_tool_output_pattern(), text, re.DOTALL))


# ----------------------------------------------------------------------
# Token-ID based helpers. Prefer these during generation: they operate
# directly on the model's output tokens so no string decoding is needed
# and partial/truncated tags are impossible (each boundary is one atomic
# vocab id). The ``tokenizer`` argument must expose
# ``convert_tokens_to_ids`` and ``decode``.
# ----------------------------------------------------------------------


def _token_id_or_none(tokenizer, token_str: str) -> Optional[int]:
    """Return the single id for ``token_str`` if the tokenizer maps it;
    otherwise ``None``. Used so callers can defensively skip token-ID
    paths when running under a tokenizer that doesn't know the new
    special tokens (e.g. an older checkpoint's persisted vocab)."""
    tid = tokenizer.convert_tokens_to_ids(token_str)
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is None or (unk is not None and tid == unk):
        return None
    return int(tid)


def tool_token_ids(tokenizer) -> Dict[str, Optional[int]]:
    """Return a dict with the ids of the four tool control tokens."""
    return {
        "call_open": _token_id_or_none(tokenizer, TOOL_CALL_OPEN),
        "call_close": _token_id_or_none(tokenizer, TOOL_CALL_CLOSE),
        "result_open": _token_id_or_none(tokenizer, TOOL_RESULT_OPEN),
        "result_close": _token_id_or_none(tokenizer, TOOL_RESULT_CLOSE),
    }


def find_unprocessed_tool_call_ids(
    token_ids: Sequence[int],
    tokenizer,
    start_position: int = 0,
) -> Optional[Tuple[Dict[str, Any], int]]:
    """Scan token IDs for the next executable tool call.

    Returns ``(tool_data, end_index)`` where ``end_index`` is the
    position immediately *after* the ``[/TOOL_CALL]`` token, suitable
    for slicing and result splicing.

    A call is considered "already processed" only if the ``[/TOOL_CALL]``
    is immediately followed by a *complete* ``[TOOL_RESULT]...[/TOOL_RESULT]``
    block. A bare ``[TOOL_RESULT]`` open with no matching close is
    treated as an in-progress hallucination and the call is still
    surfaced for execution.

    Calls with a broken bracket structure (nested ``[TOOL_CALL]`` open
    before a close) are skipped - matching restarts from the inner open
    so a later well-formed call is still discovered. Calls with a
    well-formed bracket pair but a bad body (non-JSON or undecodable)
    are surfaced with ``_malformed: True`` in the returned dict so the
    caller can splice an error ``[TOOL_RESULT]`` and prevent the model
    from fabricating one.

    ``start_position`` lets callers skip regions of the token stream
    they've already inspected.
    """
    ids = tool_token_ids(tokenizer)
    if None in (ids["call_open"], ids["call_close"]):
        return None
    call_open, call_close = ids["call_open"], ids["call_close"]
    result_open, result_close = ids["result_open"], ids["result_close"]

    i = max(0, int(start_position))
    n = len(token_ids)
    while i < n:
        if token_ids[i] != call_open:
            i += 1
            continue
        # Find this open's matching close, but bail out on a nested
        # open (which means the outer call was never closed properly).
        j = i + 1
        while j < n and token_ids[j] != call_close and token_ids[j] != call_open:
            j += 1
        if j >= n:
            return None
        if token_ids[j] == call_open:
            # Outer call is malformed; restart from the new open.
            i = j
            continue
        # If a complete [TOOL_RESULT]...[/TOOL_RESULT] block follows
        # (allowing whitespace between adjacent special tokens), treat
        # this call as already handled. A bare [TOOL_RESULT] open
        # without a matching close is the model mid-hallucinating - the
        # call still needs real execution.
        if result_open is not None and result_close is not None:
            ws = _whitespace_token_ids(tokenizer)
            k = j + 1
            while k < n and token_ids[k] in ws:
                k += 1
            if k < n and token_ids[k] == result_open:
                m = k + 1
                while m < n and token_ids[m] != result_close:
                    m += 1
                if m < n:
                    i = m + 1
                    continue
        body_ids = list(token_ids[i + 1 : j])
        try:
            body = tokenizer.decode(body_ids, skip_special_tokens=False)
        except Exception as exc:
            return (
                {"_malformed": True, "_error": f"could not decode call body: {exc}"},
                j + 1,
            )
        try:
            tool_data = json.loads(body.strip())
        except json.JSONDecodeError as exc:
            return (
                {
                    "_malformed": True,
                    "_error": f"tool call body is not valid JSON: {exc.msg}",
                    "_body": body.strip(),
                },
                j + 1,
            )
        return (tool_data, j + 1)
    return None


def zero_tool_result_regions(token_ids, assistant_mask, tokenizer):
    """Zero ``assistant_mask`` over every ``[TOOL_RESULT]...[/TOOL_RESULT]``
    span (markers included).

    Tool results are runtime-injected at inference, so any loss spent on
    predicting their tokens is wasted at best and trains the model to
    fabricate plausible-looking results at worst. The open and close
    markers are also injected, so the masked range covers both.

    Returns the mask unchanged when the tokenizer doesn't expose the
    tool-result special tokens (older checkpoints). On a span with no
    matching close (malformed / truncated mid-doc), the range is zeroed
    from the open through end-of-sequence as a defensive measure.
    """
    ids = tool_token_ids(tokenizer)
    open_id, close_id = ids["result_open"], ids["result_close"]
    if open_id is None or close_id is None:
        return assistant_mask

    if hasattr(token_ids, "tolist"):
        ids_list = token_ids.tolist()
    else:
        ids_list = list(token_ids)

    out = assistant_mask.clone() if hasattr(assistant_mask, "clone") else list(assistant_mask)
    n = len(ids_list)
    i = 0
    while i < n:
        if ids_list[i] != open_id:
            i += 1
            continue
        j = i + 1
        while j < n and ids_list[j] != close_id:
            j += 1
        end = j if j < n else n - 1
        if hasattr(out, "clone"):
            out[i : end + 1] = 0
        else:
            for k in range(i, end + 1):
                out[k] = 0
        i = end + 1
    return out


def has_complete_tool_call_ids(token_ids: Sequence[int], tokenizer) -> bool:
    """True iff the token stream contains a complete ``[TOOL_CALL]...[/TOOL_CALL]``."""
    ids = tool_token_ids(tokenizer)
    call_open, call_close = ids["call_open"], ids["call_close"]
    if call_open is None or call_close is None:
        return False
    try:
        start = token_ids.index(call_open)
    except ValueError:
        return False
    rest = token_ids[start + 1 :]
    return call_close in rest


def has_tool_output_ids(token_ids: Sequence[int], tokenizer) -> bool:
    """True iff the token stream contains a complete ``[TOOL_RESULT]...[/TOOL_RESULT]``."""
    ids = tool_token_ids(tokenizer)
    result_open, result_close = ids["result_open"], ids["result_close"]
    if result_open is None or result_close is None:
        return False
    try:
        start = token_ids.index(result_open)
    except ValueError:
        return False
    rest = token_ids[start + 1 :]
    return result_close in rest


def build_result_splice_ids(tokenizer, result: Any) -> List[int]:
    """Encode ``\\n[TOOL_RESULT]\\n<result>\\n[/TOOL_RESULT]`` as a list of
    token ids.

    The leading newline separates the splice from the preceding
    ``[/TOOL_CALL]``; the inner newlines keep the markers on their own
    lines around the body. Open/close markers are single atomic tokens;
    the body and surrounding whitespace are encoded as ordinary text.
    """
    result_open_id = _token_id_or_none(tokenizer, TOOL_RESULT_OPEN)
    result_close_id = _token_id_or_none(tokenizer, TOOL_RESULT_CLOSE)
    if result_open_id is None or result_close_id is None:
        return list(
            tokenizer.encode("\n" + format_tool_output(result), add_special_tokens=False)
        )
    leading_ids = list(tokenizer.encode("\n", add_special_tokens=False))
    body_ids = list(tokenizer.encode(f"\n{result}\n", add_special_tokens=False))
    return [*leading_ids, result_open_id, *body_ids, result_close_id]


def _whitespace_token_ids(tokenizer) -> set:
    """Token ids for common whitespace characters, used by the parser
    to tolerate ``\\n`` / spaces between adjacent special tokens."""
    ids: set = set()
    for ch in ("\n", " ", "\t", "\r"):
        try:
            ids.update(tokenizer.encode(ch, add_special_tokens=False))
        except Exception:
            pass
    return ids
