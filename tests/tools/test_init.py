import pytest

from praxis.tools import (
    ToolValidationError,
    calc,
    call_tool,
    get_tools,
    validate_tool_arguments,
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


def test_execute_tool_call_guards_non_dict():
    """execute_tool_call must not crash on a non-dict (defense in depth for the
    'int object has no attribute get' regression)."""
    from praxis.tools import execute_tool_call

    for bad in (5, "hi", [1, 2], None):
        result = execute_tool_call(bad, [])
        assert isinstance(result, str) and result.startswith("Error:")


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


# ------------------------------------------------------------------------------
# tool_reporting
# ------------------------------------------------------------------------------
# Telling the client that a tool ran.
#
# The reply is the wrong place to look. Under either chat format the extractor strips
# the whole call/result exchange out of it (see ``praxis.generation.reply``), and under
# ``tool_style="roles"`` the streamer is muted for the duration of the call on top of
# that - so a turn that consulted a tool and a turn that made the same claim up produce
# byte-identical text. The only account of the difference is ``on_tool``, fired from the
# branch that executes the tool, and these are its properties:
#
# - it names the tool that actually RAN, resolved the same way the executor resolves it,
# - a call that never ran is never announced, - and the announcement survives the reset
# that follows it, because the reset retracts the model's pre-call chatter and not the
# fact of the call.


def test_the_reported_name_is_the_one_the_executor_resolved(tokenizer):
    """Three spellings reach the executor (``name``, ``tool``, and OpenAI's
    nested ``function.name``). The badge shares its resolver, so it cannot name
    a different tool than the one that ran."""
    from praxis.tools import tool_call_name

    for call in (
        {"name": "calc"},
        {"tool": "calc"},
        {"function": {"name": "calc"}},
    ):
        assert tool_call_name(call) == "calc"
    assert tool_call_name({"arguments": {}}) is None
    assert tool_call_name("not a dict") is None
