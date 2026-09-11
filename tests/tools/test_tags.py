import json

import pytest

from praxis.tokenizers.byte_level import ByteLevelTokenizer
from praxis.tokenizers.chat_templates import chat_format_of
from praxis.tools import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESULT_CLOSE,
    TOOL_RESULT_OPEN,
    build_result_splice_ids,
    build_result_splice_text,
    calc,
    classify_boundary_halt,
    find_pending_call_text,
    find_unprocessed_tool_call_ids,
    format_call_body,
    format_tool_call_with_result,
    format_tool_input,
    format_tool_output,
    get_unprocessed_tool_call,
    has_complete_tool_call,
    has_complete_tool_call_ids,
    has_tool_output,
    has_tool_output_ids,
    parse_tool_call,
    tool_token_ids,
)

# ------------------------------------------------------------------------------
# tools
# ------------------------------------------------------------------------------
# Tool-calling tests.
#
# Tool-call boundaries are atomic special tokens
# (``[TOOL_CALL]``/``[/TOOL_CALL]``/``[TOOL_RESULT]``/``[/TOOL_RESULT]``). The tests
# exercise both the string-form helpers (format, parse, regex patterns) and the token-ID
# helpers used by the generator at runtime.


# ---------------------------------------------------------------------------
# String-form format/parse helpers.
# ---------------------------------------------------------------------------


def test_format_tool_input_uses_atomic_strings():
    out = format_tool_input("calc", {"values": [1, 2], "op": "add"})
    assert out.startswith(TOOL_CALL_OPEN)
    assert out.rstrip().endswith(TOOL_CALL_CLOSE)
    assert '"name": "calc"' in out
    # Legacy inline string forms must be gone.
    assert "<tin>" not in out and "</tin>" not in out


def test_format_tool_output_uses_atomic_strings():
    out = format_tool_output(42)
    assert out == f"{TOOL_RESULT_OPEN}\n42\n{TOOL_RESULT_CLOSE}"
    assert "<tout>" not in out


def test_format_tool_call_with_result_separates_special_tokens():
    s = format_tool_call_with_result("calc", {"values": [1, 2], "op": "add"}, 3)
    # Adjacent special tokens must be separated by whitespace, never butted up.
    assert TOOL_CALL_CLOSE + TOOL_RESULT_OPEN not in s
    assert f"{TOOL_CALL_CLOSE}\n{TOOL_RESULT_OPEN}" in s
    assert f"{TOOL_RESULT_OPEN}\n3\n{TOOL_RESULT_CLOSE}" in s


def test_parse_tool_call_extracts_last_valid():
    text = format_tool_input("calc", {"values": [10, 20], "op": "add"})
    parsed = parse_tool_call(text)
    assert parsed is not None
    assert parsed["name"] == "calc"
    assert parsed["arguments"] == {"values": [10, 20], "op": "add"}


def test_has_complete_and_has_output_text_helpers():
    call_only = format_tool_input("calc", {"values": [1, 1], "op": "add"})
    complete = format_tool_call_with_result("calc", {"values": [1, 1], "op": "add"}, 2)
    assert has_complete_tool_call(call_only) is True
    assert has_tool_output(call_only) is False
    assert has_complete_tool_call(complete) is True
    assert has_tool_output(complete) is True
    assert has_complete_tool_call("no tool here") is False


def test_get_unprocessed_tool_call_text_helper():
    call_only = format_tool_input("calc", {"values": [1, 1], "op": "add"})
    result = get_unprocessed_tool_call(call_only)
    assert result is not None and result[0]["name"] == "calc"

    resolved = format_tool_call_with_result("calc", {"values": [1, 1], "op": "add"}, 2)
    assert get_unprocessed_tool_call(resolved) is None


def test_multiple_calls_unprocessed_scan_order():
    # First call resolved, second pending: should surface the second.
    first = format_tool_call_with_result("calc", {"values": [1, 1], "op": "add"}, 2)
    second_open = format_tool_input("calc", {"values": [3, 4], "op": "mul"})
    text = f"{first}\n{second_open}"
    got = get_unprocessed_tool_call(text)
    assert got is not None
    assert got[0]["arguments"] == {"values": [3, 4], "op": "mul"}


def test_tool_token_ids_lookup_returns_ints():
    tok = ByteLevelTokenizer()
    mapping = tool_token_ids(tok)
    for key in ("call_open", "call_close", "result_open", "result_close"):
        assert isinstance(mapping[key], int)
    # All four ids must be distinct.
    assert len(set(mapping.values())) == 4


def test_find_unprocessed_tool_call_ids_locates_pending_call():
    tok = ByteLevelTokenizer()
    s = format_tool_input("calc", {"values": [1, 2], "op": "add"})
    ids = list(tok.encode(s, add_special_tokens=False))
    found = find_unprocessed_tool_call_ids(ids, tok)
    assert found is not None
    call, end_idx = found
    assert call == {"name": "calc", "arguments": {"values": [1, 2], "op": "add"}}
    # end_idx points one past the close token.
    assert ids[end_idx - 1] == tok.tool_call_end_token_id


def test_find_unprocessed_tool_call_ids_skips_resolved_calls():
    """A call followed by a *complete* [TOOL_RESULT]...[/TOOL_RESULT]
    block is treated as already handled - eos halting at [/TOOL_CALL]
    means the model never gets to fully complete a result block before
    we splice ours in, so a complete block is reliably ours."""
    tok = ByteLevelTokenizer()
    resolved = format_tool_call_with_result("calc", {"values": [1, 1], "op": "add"}, 2)
    ids = list(tok.encode(resolved, add_special_tokens=False))
    assert find_unprocessed_tool_call_ids(ids, tok) is None


def test_find_unprocessed_tool_call_ids_returns_none_on_no_tool():
    tok = ByteLevelTokenizer()
    ids = list(tok.encode("just some chatter", add_special_tokens=False))
    assert find_unprocessed_tool_call_ids(ids, tok) is None


def test_non_object_tool_body_is_malformed_not_fatal():
    """A bare JSON value (e.g. "5") parses fine but isn't a tool call object;
    it must be flagged malformed, never returned as a raw int that downstream
    .get() would crash on."""
    tok = ByteLevelTokenizer()
    open_id = tok.tool_call_token_id
    close_id = tok.tool_call_end_token_id
    body = list(tok.encode("5", add_special_tokens=False))
    ids = [open_id] + body + [close_id]
    found = find_unprocessed_tool_call_ids(ids, tok)
    assert found is not None
    call, _ = found
    assert isinstance(call, dict) and call.get("_malformed")


def test_has_complete_tool_call_and_output_ids():
    tok = ByteLevelTokenizer()
    open_only = list(
        tok.encode(
            format_tool_input("calc", {"values": [1], "op": "add"}),
            add_special_tokens=False,
        )
    )
    complete = list(
        tok.encode(
            format_tool_call_with_result("calc", {"values": [1], "op": "add"}, 1),
            add_special_tokens=False,
        )
    )
    assert has_complete_tool_call_ids(open_only, tok) is True
    assert has_tool_output_ids(open_only, tok) is False
    assert has_complete_tool_call_ids(complete, tok) is True
    assert has_tool_output_ids(complete, tok) is True


def test_build_result_splice_ids_separates_markers_with_newlines():
    tok = ByteLevelTokenizer()
    spliced = build_result_splice_ids(tok, 42)
    # The splice mirrors the chat-template's multi-turn structure:
    #   \n[SEP]\n[BOS]tool\n[TOOL_RESULT]\n42\n[/TOOL_RESULT]\n[SEP]\n[BOS]assistant\n
    # The result markers stay atomic, the body is wrapped in newlines,
    # and the splice ends with a fresh [BOS]assistant role transition
    # so the model is in the same context it was supervised in.
    assert tok.tool_result_token_id in spliced
    assert tok.tool_result_end_token_id in spliced

    open_pos = spliced.index(tok.tool_result_token_id)
    close_pos = spliced.index(tok.tool_result_end_token_id)
    body_text = tok.decode(spliced[open_pos + 1 : close_pos], skip_special_tokens=True)
    assert body_text.strip() == "42"
    assert body_text.startswith("\n") and body_text.endswith("\n")

    # SEP appears both before and after the tool turn; BOS appears for
    # both the tool role and the trailing assistant role.
    assert spliced.count(tok.sep_token_id) == 2
    assert spliced.count(tok.bos_token_id) == 2

    # The trailing role marker is `assistant`, so the model picks up in
    # the right turn for its natural-language follow-up.
    tail = tok.decode(spliced[-len(b"assistant\n") :], skip_special_tokens=True)
    assert tail.strip() == "assistant"


def test_roundtrip_splice_preserves_pending_call_detection():
    tok = ByteLevelTokenizer()
    # Start with a pending tool call.
    call_text = format_tool_input("calc", {"values": [5, 5], "op": "mul"})
    ids = list(tok.encode(call_text, add_special_tokens=False))
    found = find_unprocessed_tool_call_ids(ids, tok)
    assert found is not None
    call, end_idx = found

    # Splice the result in; the call is now resolved.
    result_ids = build_result_splice_ids(tok, 25)
    spliced = ids[:end_idx] + result_ids + ids[end_idx:]
    assert find_unprocessed_tool_call_ids(spliced, tok) is None


def test_find_unprocessed_skips_malformed_call_then_finds_valid_one():
    """A malformed earlier call must not swallow a later valid one.

    Regression for the case where the model emits a broken
    [TOOL_CALL]...[/TOOL_RESULT] (wrong close), then later emits a
    well-formed [TOOL_CALL]...[/TOOL_CALL]. The parser used to greedily
    extend the first open's match all the way to the second close, fail
    JSON parsing on the merged body, and skip past both calls.
    """
    tok = ByteLevelTokenizer()
    malformed = f"{TOOL_CALL_OPEN}\nnot json at all{TOOL_RESULT_CLOSE}\n"
    valid = format_tool_input("calc", {"values": [4971, 242], "op": "div"})
    ids = list(tok.encode(malformed + valid, add_special_tokens=False))
    found = find_unprocessed_tool_call_ids(ids, tok)
    assert found is not None
    call, _ = found
    assert call["name"] == "calc"
    assert call["arguments"]["op"] == "div"


def test_find_unprocessed_executes_call_with_partial_hallucinated_result():
    """A bare [TOOL_RESULT] open with no matching close is a model
    mid-hallucination - the call should still be surfaced for execution."""
    tok = ByteLevelTokenizer()
    call_text = format_tool_input("calc", {"values": [3, 4], "op": "mul"})
    # Model started hallucinating a result but never closed it.
    partial = call_text + TOOL_RESULT_OPEN + "12"
    ids = list(tok.encode(partial, add_special_tokens=False))
    found = find_unprocessed_tool_call_ids(ids, tok)
    assert found is not None
    call, _ = found
    assert call["name"] == "calc"


def test_find_unprocessed_returns_malformed_for_bad_json_body():
    """Bad JSON inside a well-formed bracket pair must surface as a
    ``_malformed`` sentinel so the generator can splice an error result
    instead of letting the model hallucinate one."""
    tok = ByteLevelTokenizer()
    bad = f"{TOOL_CALL_OPEN}\nnot valid json\n{TOOL_CALL_CLOSE}"
    ids = list(tok.encode(bad, add_special_tokens=False))
    found = find_unprocessed_tool_call_ids(ids, tok)
    assert found is not None
    call, _ = found
    assert call.get("_malformed") is True
    assert "JSON" in call.get("_error", "")


# ------------------------------------------------------------------------------
# chat_formats
# ------------------------------------------------------------------------------
# Tests for the ``chat_formats`` registry and the text-boundary (prose) format.
#
# The invariants worth pinning are the ones that silently produce a broken run rather
# than an exception:
#
# - the `default` profile must stay byte-identical, since every existing checkpoint's
# data pipeline depends on it, - the boundary that ENDS a generated turn must be a
# trained target (the defect `prose` exists to remove), - a stop-string halt must not
# re-fire on the boundary it resumed from, or the tool loop returns zero new tokens
# forever, - the tool flow's three boundaries must classify unambiguously.


# ------------------------------------------------------------- tool flow


def test_prose_tool_boundaries_classify(prose_tokenizer):
    fmt = chat_format_of(prose_tokenizer)
    call_open = "user\n\n2+2?\n\nassistant\n\nlet me check.\n\ncall\n\n"
    assert classify_boundary_halt(call_open, fmt) == "call_open"

    body = format_call_body("calc", {"values": [2, 2], "op": "add"})
    call_close = call_open + body + "\n\ntool\n\n"
    assert classify_boundary_halt(call_close, fmt) == "call_close"

    # A plain turn terminator is neither; the caller treats it as done.
    assert classify_boundary_halt(call_open + body + "\n\nuser\n\n", fmt) is None


def test_prose_pending_call_round_trip(prose_tokenizer):
    fmt = chat_format_of(prose_tokenizer)
    body = format_call_body("calc", {"values": [750, 485], "op": "mul"})
    text = f"assistant\n\nok\n\ncall\n\n{body}\n\ntool\n\n"
    assert find_pending_call_text(text, fmt) == {
        "name": "calc",
        "arguments": {"values": [750, 485], "op": "mul"},
    }
    assert build_result_splice_text("363750.0", fmt) == "363750.0\n\nassistant\n\n"


def test_prose_malformed_call_is_surfaced_not_guessed(prose_tokenizer):
    fmt = chat_format_of(prose_tokenizer)
    call = find_pending_call_text("call\n\nnot json at all\n\ntool\n\n", fmt)
    assert call is not None and call["_malformed"] is True

    # A JSON scalar is valid JSON but cannot be a call.
    scalar = find_pending_call_text("call\n\n5\n\ntool\n\n", fmt)
    assert scalar is not None and scalar["_malformed"] is True


def test_pending_call_requires_the_result_boundary(prose_tokenizer):
    fmt = chat_format_of(prose_tokenizer)
    assert find_pending_call_text("call\n\n{}\n\n", fmt) is None
    assert find_pending_call_text("assistant\n\nhi\n\ntool\n\n", fmt) is None


def test_answered_call_is_not_pending_again(prose_tokenizer):
    """A spliced result ends the call. Re-finding it fabricates an error.

    Halting on the result boundary a second time in one request would rfind the
    same `call` turn and hand json.loads its body PLUS the spliced result and
    the reply that followed - which fails to parse, so an error result gets
    spliced over a turn that was already answered correctly.
    """
    fmt = chat_format_of(prose_tokenizer)
    answered = (
        'user\n\nWhat is 2+2?\n\nassistant\n\n\n\ncall\n\n{"name": "calc", '
        '"arguments": {}}\n\ntool\n\n4\n\nassistant\n\nIt is 4.\n\ntool\n\n'
    )
    assert find_pending_call_text(answered, fmt) is None
