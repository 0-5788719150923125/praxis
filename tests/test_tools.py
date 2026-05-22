"""Tool-calling tests.

Tool-call boundaries are atomic special tokens
(``[TOOL_CALL]``/``[/TOOL_CALL]``/``[TOOL_RESULT]``/``[/TOOL_RESULT]``).
The tests exercise both the string-form helpers (format, parse, regex
patterns) and the token-ID helpers used by the generator at runtime.
"""

import json
from unittest.mock import MagicMock

import pytest
import torch

from praxis.tokenizers.byte_level import ByteLevelTokenizer
from praxis.tokenizers.char_level import CharLevelTokenizer
from praxis.tools import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESULT_CLOSE,
    TOOL_RESULT_OPEN,
    ToolValidationError,
    build_result_splice_ids,
    calc,
    call_tool,
    find_unprocessed_tool_call_ids,
    format_tool_call_with_result,
    format_tool_input,
    format_tool_output,
    get_tools,
    get_unprocessed_tool_call,
    has_complete_tool_call,
    has_complete_tool_call_ids,
    has_tool_output,
    has_tool_output_ids,
    parse_tool_call,
    tool_token_ids,
    validate_tool_arguments,
)


# ---------------------------------------------------------------------------
# Tool functions themselves (unchanged by the conversion).
# ---------------------------------------------------------------------------


def test_calc_basic_ops():
    assert calc.forward(values=[10, 20, 30], op="add") == 60
    assert calc.forward(values=[100, 25], op="sub") == 75
    assert calc.forward(values=[5, 6], op="mul") == 30.0
    assert calc.forward(values=[50, 2], op="div") == 25.0
    assert calc.forward(values=[9], op="sqrt") == 3.0
    assert calc.forward(values=[2, 3], op="exp") == 8.0


def test_calc_error_handling():
    with pytest.raises(ValueError, match="Division by zero"):
        calc.forward(values=[10, 0], op="div")
    with pytest.raises(ValueError, match="negative number"):
        calc.forward(values=[-4], op="sqrt")
    with pytest.raises(ValueError, match="values list cannot be empty"):
        calc.forward(values=[], op="add")
    with pytest.raises(ValueError, match="Unknown operation"):
        calc.forward(values=[1, 2], op="unknown_op")


def test_get_tools_registry():
    tools = get_tools()
    assert len(tools) >= 1
    names = [t.name for t in tools]
    assert "calc" in names


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
    s = format_tool_call_with_result(
        "calc", {"values": [1, 2], "op": "add"}, 3
    )
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
    complete = format_tool_call_with_result(
        "calc", {"values": [1, 1], "op": "add"}, 2
    )
    assert has_complete_tool_call(call_only) is True
    assert has_tool_output(call_only) is False
    assert has_complete_tool_call(complete) is True
    assert has_tool_output(complete) is True
    assert has_complete_tool_call("no tool here") is False


def test_get_unprocessed_tool_call_text_helper():
    call_only = format_tool_input("calc", {"values": [1, 1], "op": "add"})
    result = get_unprocessed_tool_call(call_only)
    assert result is not None and result[0]["name"] == "calc"

    resolved = format_tool_call_with_result(
        "calc", {"values": [1, 1], "op": "add"}, 2
    )
    assert get_unprocessed_tool_call(resolved) is None


def test_multiple_calls_unprocessed_scan_order():
    # First call resolved, second pending: should surface the second.
    first = format_tool_call_with_result(
        "calc", {"values": [1, 1], "op": "add"}, 2
    )
    second_open = format_tool_input("calc", {"values": [3, 4], "op": "mul"})
    text = f"{first}\n{second_open}"
    got = get_unprocessed_tool_call(text)
    assert got is not None
    assert got[0]["arguments"] == {"values": [3, 4], "op": "mul"}


def test_regex_patterns_escape_brackets_correctly():
    # Both literal '[' characters must be escaped for the regex to match.
    from praxis.tools.tags import get_tool_input_pattern, get_tool_output_pattern

    import re

    tin = format_tool_input("calc", {"values": [1], "op": "add"})
    assert re.search(get_tool_input_pattern(), tin, re.DOTALL) is not None

    tout = format_tool_output(7)
    assert re.search(get_tool_output_pattern(), tout, re.DOTALL) is not None


# ---------------------------------------------------------------------------
# Token-ID helpers - the generator's runtime path.
# ---------------------------------------------------------------------------


def test_tokenizer_encodes_tool_tokens_as_single_ids_byte_level():
    tok = ByteLevelTokenizer()
    for marker in (TOOL_CALL_OPEN, TOOL_CALL_CLOSE, TOOL_RESULT_OPEN, TOOL_RESULT_CLOSE):
        ids = tok.encode(marker, add_special_tokens=False)
        assert len(ids) == 1, f"{marker} not atomic: {ids}"


def test_tokenizer_encodes_tool_tokens_as_single_ids_char_level():
    tok = CharLevelTokenizer()
    for marker in (TOOL_CALL_OPEN, TOOL_CALL_CLOSE, TOOL_RESULT_OPEN, TOOL_RESULT_CLOSE):
        ids = tok.encode(marker, add_special_tokens=False)
        assert len(ids) == 1, f"{marker} not atomic: {ids}"


def test_tool_token_ids_lookup_returns_ints():
    tok = ByteLevelTokenizer()
    mapping = tool_token_ids(tok)
    for key in ("call_open", "call_close", "result_open", "result_close"):
        assert isinstance(mapping[key], int)
    # All four ids must be distinct.
    assert len(set(mapping.values())) == 4


def test_skip_special_tokens_strips_tool_markers():
    tok = ByteLevelTokenizer()
    text = f"hello {TOOL_CALL_OPEN}body{TOOL_CALL_CLOSE} done"
    ids = tok.encode(text, add_special_tokens=False)
    with_specials = tok.decode(ids, skip_special_tokens=False)
    without_specials = tok.decode(ids, skip_special_tokens=True)
    assert TOOL_CALL_OPEN in with_specials
    assert TOOL_CALL_OPEN not in without_specials


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
    resolved = format_tool_call_with_result(
        "calc", {"values": [1, 1], "op": "add"}, 2
    )
    ids = list(tok.encode(resolved, add_special_tokens=False))
    assert find_unprocessed_tool_call_ids(ids, tok) is None


def test_find_unprocessed_tool_call_ids_returns_none_on_no_tool():
    tok = ByteLevelTokenizer()
    ids = list(tok.encode("just some chatter", add_special_tokens=False))
    assert find_unprocessed_tool_call_ids(ids, tok) is None


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
            format_tool_call_with_result(
                "calc", {"values": [1], "op": "add"}, 1
            ),
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
    malformed = (
        f"{TOOL_CALL_OPEN}\nnot json at all{TOOL_RESULT_CLOSE}\n"
    )
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


def test_find_unprocessed_streaming_call_open_in_prompt():
    """In streaming, the [TOOL_CALL] open lives in the prompt and only
    [/TOOL_CALL] is freshly emitted. Scanning from position 0 must
    still find the call even though the open is far before the new
    tokens."""
    tok = ByteLevelTokenizer()
    call_text = format_tool_input("calc", {"values": [7, 8], "op": "add"})
    ids = list(tok.encode(call_text, add_special_tokens=False))
    found = find_unprocessed_tool_call_ids(ids, tok)
    assert found is not None
    call, _ = found
    assert call["arguments"]["op"] == "add"


# ---------------------------------------------------------------------------
# Generator plumbing.
# ---------------------------------------------------------------------------


def test_generator_eos_token_id_list_includes_tool_close():
    """The generator must halt on [/TOOL_CALL] via the eos list."""
    from praxis.generation.generator import Generator

    mock_model = MagicMock()
    mock_model.training = False
    mock_model.parameters.return_value = iter([torch.zeros(1)])
    tok = ByteLevelTokenizer()

    gen = Generator(mock_model, tok, device="cpu")
    eos_list = gen._eos_token_id_list()
    assert eos_list is not None
    assert tok.eos_token_id in eos_list
    assert tok.sep_token_id in eos_list
    assert tok.tool_call_end_token_id in eos_list


# ---------------------------------------------------------------------------
# Synthetic training data.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Schema validation. Guards the tool-call path against the model emitting
# calls that don't match the declared schema, so bad calls resolve to an
# error result instead of letting the model fabricate a [TOOL_RESULT].
# ---------------------------------------------------------------------------


def test_validate_accepts_valid_call():
    validate_tool_arguments("calc", {"values": [1, 2], "op": "add"})


def test_validate_accepts_missing_optional():
    # ``op`` has a default - omitting it is fine.
    validate_tool_arguments("calc", {"values": [1, 2]})


def test_validate_rejects_unknown_tool():
    with pytest.raises(ToolValidationError, match="Unknown tool"):
        validate_tool_arguments("nonexistent_tool", {})


def test_validate_rejects_missing_required():
    with pytest.raises(ToolValidationError, match="Missing required"):
        validate_tool_arguments("calc", {"op": "add"})


def test_validate_rejects_unknown_param():
    with pytest.raises(ToolValidationError, match="Unknown parameter"):
        validate_tool_arguments("calc", {"values": [1], "op": "add", "extra": 1})


def test_validate_rejects_wrong_type():
    # ``values`` must be an array, not a string.
    with pytest.raises(ToolValidationError, match="expected type 'array'"):
        validate_tool_arguments("calc", {"values": "not a list", "op": "add"})


def test_validate_rejects_non_dict_arguments():
    with pytest.raises(ToolValidationError, match="must be an object"):
        validate_tool_arguments("calc", [1, 2, 3])


def test_call_tool_raises_validation_error_on_bad_args():
    with pytest.raises(ToolValidationError):
        call_tool("calc", {"op": "add"})  # missing 'values'


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


def test_synthetic_formatter_emits_atomic_tool_boundaries_in_tokens():
    """format_tool_calling's output must produce atomic tool-token ids.

    After the tool-role refactor, the call and result live in separate
    messages: an ``assistant`` message holds ``[TOOL_CALL]...[/TOOL_CALL]``
    and a ``tool`` message holds ``[TOOL_RESULT]...[/TOOL_RESULT]``. This
    test verifies all four atomic ids are present across the conversation,
    just not within a single message anymore.
    """
    import random

    random.seed(7)
    from praxis.data.formatters.tools import format_tool_calling

    tok = ByteLevelTokenizer()
    doc = format_tool_calling({}, [], tok)

    # Find the assistant message with the tool call and the tool message
    # with the result. Earlier system / developer / user messages skip past.
    call_msg = next(
        m for m in doc["messages"]
        if m["role"] == "assistant" and TOOL_CALL_OPEN in m["content"]
    )
    result_msg = next(m for m in doc["messages"] if m["role"] == "tool")

    assert TOOL_CALL_OPEN in call_msg["content"]
    assert TOOL_CALL_CLOSE in call_msg["content"]
    assert TOOL_RESULT_OPEN in result_msg["content"]
    assert TOOL_RESULT_CLOSE in result_msg["content"]

    call_ids = tok.encode(call_msg["content"], add_special_tokens=False)
    result_ids = tok.encode(result_msg["content"], add_special_tokens=False)
    assert tok.tool_call_token_id in call_ids
    assert tok.tool_call_end_token_id in call_ids
    assert tok.tool_result_token_id in result_ids
    assert tok.tool_result_end_token_id in result_ids
