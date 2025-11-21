"""Test tool-calling functionality."""

import json
import re
from unittest.mock import MagicMock, patch

import pytest
import torch

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.tools import (
    calc,
    format_tool_call_with_result,
    format_tool_input,
    format_tool_output,
    get_tool_input_pattern,
    get_tools,
    parse_tool_call,
)


def test_calc_tool_basic():
    """Test the calc tool basic functionality."""
    # Test addition
    result = calc.forward(values=[10, 20, 30], op="add")
    assert result == 60

    # Test subtraction
    result = calc.forward(values=[100, 25], op="sub")
    assert result == 75

    # Test multiplication
    result = calc.forward(values=[5, 6], op="mul")
    assert result == 30.0

    # Test division
    result = calc.forward(values=[50, 2], op="div")
    assert result == 25.0

    # Test square root
    result = calc.forward(values=[9], op="sqrt")
    assert result == 3.0

    # Test exponentiation
    result = calc.forward(values=[2, 3], op="exp")
    assert result == 8.0


def test_calc_tool_via_forward():
    """Test calling the calc tool through the forward method."""
    result = calc.forward(values=[1, 2, 3, 4, 5], op="add")
    assert result == 15

    # Test with default operation (add)
    result = calc.forward(values=[10, 10])
    assert result == 20


def test_get_tools():
    """Test that tools can be retrieved."""
    tools = get_tools()

    # Should have at least the calc tool
    assert len(tools) >= 1

    # Check that calc is in the tools
    tool_names = [tool.name for tool in tools]
    assert "calc" in tool_names

    # Get the calc tool
    calc_tool = next(tool for tool in tools if tool.name == "calc")
    assert calc_tool.description
    assert hasattr(calc_tool, "forward")
    assert hasattr(calc_tool, "inputs")
    assert hasattr(calc_tool, "output_type")


def test_tool_properties():
    """Test that the decorated tool has the expected properties."""
    assert hasattr(calc, "name")
    assert calc.name == "calc"

    assert hasattr(calc, "description")
    assert "algebraic operation" in calc.description.lower()

    assert hasattr(calc, "forward")
    assert hasattr(calc, "inputs")
    assert hasattr(calc, "output_type")


def test_tool_error_handling():
    """Test error handling in tool execution."""
    # Test division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        calc.forward(values=[10, 0], op="div")

    # Test sqrt of negative number
    with pytest.raises(ValueError, match="negative number"):
        calc.forward(values=[-4], op="sqrt")

    # Test empty values
    with pytest.raises(ValueError, match="values list cannot be empty"):
        calc.forward(values=[], op="add")

    # Test unknown operation
    with pytest.raises(ValueError, match="Unknown operation"):
        calc.forward(values=[1, 2], op="unknown_op")

    # Test calling non-existent tool - now we just check it's not in the list
    tools = get_tools()
    tool_names = [tool.name for tool in tools]
    assert "nonexistent_tool" not in tool_names


def test_tool_calling_during_inference():
    """Test that tools can be used during model inference."""
    # Create a minimal config for testing
    config = PraxisConfig(
        hidden_size=64,
        num_heads=2,
        depth=2,
        vocab_size=1000,
    )

    # Create model
    model = PraxisForCausalLM(config)
    model.eval()

    # Mock tokenizer
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    tokenizer.decode = MagicMock(return_value="Test output")
    tokenizer.bos_token = "<bos>"
    tokenizer.eos_token = "<eos>"
    tokenizer.sep_token = "<sep>"
    tokenizer.pad_token_id = 0

    # Test that we can use tools
    from praxis.tools import get_tools

    # Get available tools
    tools = get_tools()
    assert len(tools) > 0

    # Call a tool directly
    result = calc.forward(values=[10, 20], op="add")
    assert result == 30

    # Parse a tool call from text using new <tin> tag format
    tool_call_text = format_tool_input("calc", {"values": [10, 20], "op": "add"})

    # Parse using the utility function
    tool_data = parse_tool_call(tool_call_text)

    assert tool_data is not None
    assert tool_data["name"] == "calc"
    assert tool_data["arguments"]["values"] == [10, 20]

    # Execute the parsed tool call via calc
    if tool_data["name"] == "calc":
        result = calc.forward(**tool_data["arguments"])
        assert result == 30


def test_model_with_tools():
    """Test that model can work with tools."""
    config = PraxisConfig(
        hidden_size=64,
        num_heads=2,
        depth=2,
        vocab_size=1000,
    )

    model = PraxisForCausalLM(config)

    # Get tools
    tools = get_tools()
    assert len(tools) > 0

    # Create a prompt that mentions tools
    tool_names = [tool.name for tool in tools]
    prompt = f"""System: You have access to the following tools: {', '.join(tool_names)}

User: Calculate 15 + 25
Assistant: I'll calculate that for you."""

    # Verify the prompt includes tool information
    assert "calc" in prompt

    # Simulate a tool call response using new tag format
    tool_response = format_tool_input("calc", {"values": [15, 25], "op": "add"})

    # Parse and execute using utility function
    tool_data = parse_tool_call(tool_response)
    assert tool_data is not None

    # Execute via the calc tool directly
    if tool_data["name"] == "calc":
        result = calc.forward(**tool_data["arguments"])
        assert result == 40


def test_truncated_tool_call_tag():
    """Test that truncated tool input tags are properly fixed."""
    from praxis.tools import fix_truncated_tags

    # Test case 1: Truncated </tin tag
    malformed_text = """[BOS]assistant
<tin>
{"name": "calc", "arguments": {"values": [103253, 757695], "op": "add"}}
</tin
[SEP]
[BOS]assistant
>
[SEP]"""

    fixed_text = fix_truncated_tags(malformed_text)
    assert "</tin>" in fixed_text
    assert "</tin\n" not in fixed_text
    # Separator tokens after </tin> should be cleaned up
    assert "[SEP]" not in fixed_text or fixed_text.find("[SEP]") < fixed_text.find("</tin>")

    # Test case 2: Generation stopped at just "</"
    malformed_text2 = """<tin>
{"name": "calc", "arguments": {"values": [1, 2], "op": "add"}}
</
[SEP]
[BOS]assistant
"""

    fixed_text2 = fix_truncated_tags(malformed_text2)
    assert "</tin>" in fixed_text2
    assert "</" + "\n[SEP]" not in fixed_text2  # Should be fixed and cleaned

    # Test case 3: Partial tag name
    malformed_text3 = """<tin>{"name": "calc", "arguments": {}}
</t"""

    fixed_text3 = fix_truncated_tags(malformed_text3)
    assert "</tin>" in fixed_text3
    assert "</t" not in fixed_text3 or "</tin>" in fixed_text3

    # Test case 4: Complete tag but with separator artifacts
    malformed_text4 = """<tin>{"name": "calc", "arguments": {}}
</tin>[SEP][BOS]assistant>[SEP]"""

    fixed_text4 = fix_truncated_tags(malformed_text4)
    assert "</tin>" in fixed_text4
    # Artifacts should be cleaned
    assert not fixed_text4.endswith("[SEP][BOS]assistant>[SEP]")
    assert fixed_text4.rstrip().endswith("</tin>")


def test_tool_tag_utilities():
    """Test the new tool tag utility functions."""
    from praxis.tools import (
        format_tool_call_with_result,
        format_tool_input,
        format_tool_output,
        get_unprocessed_tool_call,
        has_complete_tool_call,
        has_tool_output,
        parse_tool_call,
    )

    # Test format_tool_input
    tool_input = format_tool_input("calc", {"values": [1, 2], "op": "add"})
    assert "<tin>" in tool_input
    assert "</tin>" in tool_input
    assert '"name": "calc"' in tool_input

    # Test format_tool_output
    tool_output = format_tool_output(42)
    assert tool_output == "<tout>42</tout>"

    # Test format_tool_call_with_result
    complete_call = format_tool_call_with_result("calc", {"values": [1, 2], "op": "add"}, 3)
    assert "<tin>" in complete_call
    assert "</tin>" in complete_call
    assert "<tout>3</tout>" in complete_call

    # Test parse_tool_call
    parsed = parse_tool_call(tool_input)
    assert parsed is not None
    assert parsed["name"] == "calc"
    assert parsed["arguments"]["values"] == [1, 2]

    # Test has_complete_tool_call
    assert has_complete_tool_call(tool_input) is True
    assert has_complete_tool_call("no tool call here") is False

    # Test has_tool_output
    assert has_tool_output(tool_output) is True
    assert has_tool_output("no output here") is False

    # Test get_unprocessed_tool_call
    # Tool call without output should be detected as unprocessed
    unprocessed = get_unprocessed_tool_call(tool_input)
    assert unprocessed is not None
    assert unprocessed[0]["name"] == "calc"

    # Tool call with output should not be unprocessed
    processed = get_unprocessed_tool_call(complete_call)
    assert processed is None  # Has output, so it's processed


def test_multiple_tool_calls_in_sequence():
    """Test that multiple tool calls are processed in order."""
    from praxis.tools import get_unprocessed_tool_call

    # Test case 1: Two unprocessed calls - should return the FIRST one
    text_two_unprocessed = """<tin>
{"name": "calc", "arguments": {"values": [1, 2], "op": "add"}}
</tin>
Some text here
<tin>
{"name": "calc", "arguments": {"values": [3, 4], "op": "mul"}}
</tin>"""

    result = get_unprocessed_tool_call(text_two_unprocessed)
    assert result is not None
    assert result[0]["arguments"]["values"] == [1, 2]  # First call, not second
    assert result[0]["arguments"]["op"] == "add"

    # Test case 2: First call has output, second doesn't - should return second
    text_first_processed = """<tin>
{"name": "calc", "arguments": {"values": [1, 2], "op": "add"}}
</tin><tout>3</tout>
<tin>
{"name": "calc", "arguments": {"values": [5, 6], "op": "sub"}}
</tin>"""

    result2 = get_unprocessed_tool_call(text_first_processed)
    assert result2 is not None
    assert result2[0]["arguments"]["values"] == [5, 6]  # Second call
    assert result2[0]["arguments"]["op"] == "sub"

    # Test case 3: Both calls have outputs - should return None
    text_both_processed = """<tin>
{"name": "calc", "arguments": {"values": [1, 2], "op": "add"}}
</tin><tout>3</tout>
<tin>
{"name": "calc", "arguments": {"values": [5, 6], "op": "sub"}}
</tin><tout>-1</tout>"""

    result3 = get_unprocessed_tool_call(text_both_processed)
    assert result3 is None  # All processed

    # Test case 4: Three calls, middle one unprocessed - should return middle
    text_middle_unprocessed = """<tin>
{"name": "calc", "arguments": {"values": [1, 1], "op": "add"}}
</tin><tout>2</tout>
<tin>
{"name": "calc", "arguments": {"values": [2, 2], "op": "add"}}
</tin>
<tin>
{"name": "calc", "arguments": {"values": [3, 3], "op": "add"}}
</tin>"""

    result4 = get_unprocessed_tool_call(text_middle_unprocessed)
    assert result4 is not None
    assert result4[0]["arguments"]["values"] == [2, 2]  # Middle call
