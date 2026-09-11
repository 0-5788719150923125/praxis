"""``praxis.tools``: the tools themselves, schema validation, and execution."""

import pytest

from praxis.tools import (
    ToolValidationError,
    calc,
    call_tool,
    execute_tool_call,
    get_tools,
    tool_call_name,
    validate_tool_arguments,
)


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
    for bad in (5, "hi", [1, 2], None):
        result = execute_tool_call(bad, [])
        assert isinstance(result, str) and result.startswith("Error:")


# ---------------------------------------------------------------------------
# Schema validation. Guards the tool-call path against the model emitting
# calls that don't match the declared schema, so bad calls resolve to an
# error result instead of letting the model fabricate a [TOOL_RESULT].
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {"values": [1, 2], "op": "add"},
        {"values": [1, 2]},  # ``op`` has a default
    ],
)
def test_validate_accepts_valid_calls(arguments):
    validate_tool_arguments("calc", arguments)


@pytest.mark.parametrize(
    "name,arguments,match",
    [
        ("nonexistent_tool", {}, "Unknown tool"),
        ("calc", {"op": "add"}, "Missing required"),
        ("calc", {"values": [1], "op": "add", "extra": 1}, "Unknown parameter"),
        ("calc", {"values": "not a list", "op": "add"}, "expected type 'array'"),
        ("calc", [1, 2, 3], "must be an object"),
    ],
)
def test_validate_rejects_bad_calls(name, arguments, match):
    with pytest.raises(ToolValidationError, match=match):
        validate_tool_arguments(name, arguments)


def test_call_tool_raises_validation_error_on_bad_args():
    with pytest.raises(ToolValidationError):
        call_tool("calc", {"op": "add"})  # missing 'values'


def test_the_reported_name_is_the_one_the_executor_resolved():
    """Three spellings reach the executor (``name``, ``tool``, and OpenAI's
    nested ``function.name``). The badge shares its resolver, so it cannot name
    a different tool than the one that ran."""
    for call in (
        {"name": "calc"},
        {"tool": "calc"},
        {"function": {"name": "calc"}},
    ):
        assert tool_call_name(call) == "calc"
    assert tool_call_name({"arguments": {}}) is None
    assert tool_call_name("not a dict") is None
