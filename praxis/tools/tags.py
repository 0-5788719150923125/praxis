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


def format_call_body(
    tool_name: str, arguments: Dict[str, Any], indent: Optional[int] = None
) -> str:
    """The bare call JSON, with no delimiters of any kind.

    This is the payload both tool styles carry: ``tool_style="tokens"`` wraps
    it in ``[TOOL_CALL]``/``[/TOOL_CALL]``, while ``tool_style="roles"`` puts
    it in a turn of its own where the surrounding role boundaries already
    delimit it.

    Example:
        >>> format_call_body("calc", {"values": [1, 2], "op": "add"})
        '{"name": "calc", "arguments": {"values": [1, 2], "op": "add"}}'
    """
    return json.dumps({"name": tool_name, "arguments": arguments}, indent=indent)


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
    json_str = format_call_body(tool_name, arguments, indent)
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
        # before the next [TOOL_CALL], treat this call as already
        # handled. The splice emitted by build_result_splice_ids puts
        # SEP/BOS/role-name tokens between [/TOOL_CALL] and
        # [TOOL_RESULT] to match the chat template, so the gap is no
        # longer whitespace-only - scan past any non-special tokens
        # until we hit either the result open or a new call open. A
        # bare [TOOL_RESULT] open without a matching close is the
        # model mid-hallucinating - the call still needs real
        # execution.
        if result_open is not None and result_close is not None:
            k = j + 1
            while k < n and token_ids[k] != result_open and token_ids[k] != call_open:
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
        # Valid JSON, but a call must be an object - the model can emit a bare
        # int/string/list (e.g. "5"), which would break dict-based handling.
        if not isinstance(tool_data, dict):
            return (
                {
                    "_malformed": True,
                    "_error": "tool call body must be a JSON object, got "
                    f"{type(tool_data).__name__}",
                    "_body": body.strip(),
                },
                j + 1,
            )
        return (tool_data, j + 1)
    return None


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


# ----------------------------------------------------------------------
# Text-boundary tool flow (``tool_style="roles"``). The call and its result
# are turns in their own right, so the role boundaries that open them ARE the
# delimiters - there is nothing left for a control token to mark:
#
#     ...\\n\\ncall\\n\\n{"name": ...}\\n\\ntool\\n\\n<result>\\n\\nassistant\\n\\n...
#
# The generator's three halt cases map across unchanged: the `call` boundary
# opens the JSON (switch to greedy), the `result` boundary closes it (execute
# and splice), any other role's boundary ends the turn.
# ----------------------------------------------------------------------


def classify_boundary_halt(text: str, fmt) -> Optional[str]:
    """Which tool-flow boundary ``text`` ends on: ``"call_open"``,
    ``"call_close"``, or ``None`` for anything else (including a plain
    end-of-turn boundary, which the caller treats as done).
    """
    if fmt.tool_style != "roles":
        return None
    if text.endswith(fmt.boundary(fmt.call_role)):
        return "call_open"
    if text.endswith(fmt.boundary(fmt.result_role)):
        return "call_close"
    return None


def find_pending_call_text(text: str, fmt) -> Optional[Dict[str, Any]]:
    """Parse the call whose result boundary ``text`` just landed on.

    Only called when generation halted on the result boundary, so the pending
    call is unambiguously the last one: everything between the most recent
    ``call`` boundary and the trailing ``result`` boundary. Returns ``None``
    when there is no call to answer, or a dict with ``_malformed: True`` when
    the body is not a JSON object - the caller splices an error result rather
    than letting the model fabricate one.
    """
    call_boundary = fmt.boundary(fmt.call_role)
    result_boundary = fmt.boundary(fmt.result_role)
    if not text.endswith(result_boundary):
        return None
    body_end = len(text) - len(result_boundary)
    start = text.rfind(call_boundary, 0, body_end)
    if start != -1:
        body_start = start + len(call_boundary)
    else:
        # A prompt long enough to be left-truncated (``_prepare_inputs`` caps it
        # against max_position_embeddings) can lose the boundary's leading blank
        # line, leaving the bare role line at position 0. The call is still
        # executable, so recognise that form rather than dropping the call and
        # letting the model invent its own result.
        bare = call_boundary.lstrip("\n")
        if not text.startswith(bare):
            return None
        body_start = len(bare)
    body = text[body_start:body_end].strip()
    try:
        tool_data = json.loads(body)
    except json.JSONDecodeError as exc:
        return {
            "_malformed": True,
            "_error": f"tool call body is not valid JSON: {exc.msg}",
            "_body": body,
        }
    if not isinstance(tool_data, dict):
        return {
            "_malformed": True,
            "_error": "tool call body must be a JSON object, got "
            f"{type(tool_data).__name__}",
            "_body": body,
        }
    return tool_data


def build_result_splice_text(result: Any, fmt) -> str:
    """The text spliced after a ``result`` boundary: the real result, then the
    boundary handing control back to the model.

    Generation halted with the sequence ending in ``\\n\\ntool\\n\\n``, so the
    result body follows directly and the reply boundary closes the tool turn -
    byte-identical to what the training template renders for the same
    exchange, which is the point (the model has to be IN distribution when it
    resumes, or it stops mid-turn).
    """
    return f"{result}{fmt.boundary(fmt.reply_role)}"


def build_result_splice_ids(tokenizer, result: Any) -> List[int]:
    """Encode the post-tool-call splice as a list of token ids.

    The splice mirrors the multi-turn structure the chat template
    produces at training time so inference stays in-distribution:

        \\n[SEP]\\n[BOS]tool\\n
        [TOOL_RESULT]\\n<result>\\n[/TOOL_RESULT]
        \\n[SEP]\\n[BOS]assistant\\n

    Matching the training format here is critical: the original inline
    splice left the model in a context it had never been supervised in,
    so it tended to emit EOS instead of a natural-language reply.

    Text-boundary formats take the role-boundary form above instead.

    Falls back to a plain-text encoding when the tokenizer is missing
    any of the required special-token ids (e.g. an older checkpoint).
    """
    from praxis.tokenizers.chat_templates import chat_format_of

    fmt = chat_format_of(tokenizer)
    if fmt.tool_style == "roles":
        return list(
            tokenizer.encode(
                build_result_splice_text(result, fmt), add_special_tokens=False
            )
        )

    result_open_id = _token_id_or_none(tokenizer, TOOL_RESULT_OPEN)
    result_close_id = _token_id_or_none(tokenizer, TOOL_RESULT_CLOSE)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if None in (result_open_id, result_close_id, sep_id, bos_id):
        return list(
            tokenizer.encode(
                "\n" + format_tool_output(result), add_special_tokens=False
            )
        )

    nl_ids = list(tokenizer.encode("\n", add_special_tokens=False))
    tool_role_ids = list(tokenizer.encode("tool", add_special_tokens=False))
    assistant_role_ids = list(tokenizer.encode("assistant", add_special_tokens=False))
    body_ids = list(tokenizer.encode(f"\n{result}\n", add_special_tokens=False))

    return [
        *nl_ids,
        int(sep_id),
        *nl_ids,
        int(bos_id),
        *tool_role_ids,
        *nl_ids,
        result_open_id,
        *body_ids,
        result_close_id,
        *nl_ids,
        int(sep_id),
        *nl_ids,
        int(bos_id),
        *assistant_role_ids,
        *nl_ids,
    ]


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
