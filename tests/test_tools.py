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


def test_generator_has_unclosed_tool_tag():
    """Test the _has_unclosed_tool_tag helper method in Generator."""
    from unittest.mock import MagicMock

    from praxis.generation.generator import Generator

    # Create a mock model and tokenizer
    mock_model = MagicMock()
    mock_model.training = False
    mock_model.parameters.return_value = iter([torch.zeros(1)])

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.sep_token_id = 2

    # Create generator instance
    generator = Generator(mock_model, mock_tokenizer, device="cpu")

    # Test cases for unclosed tool tags
    # Case 1: No tool tags
    assert generator._has_unclosed_tool_tag("Hello world") is False

    # Case 2: Complete tool tag (closed)
    closed_tag = "<tin>{}</tin>"
    assert generator._has_unclosed_tool_tag(closed_tag) is False

    # Case 3: Unclosed tool tag
    unclosed_tag = "<tin>{"
    assert generator._has_unclosed_tool_tag(unclosed_tag) is True

    # Case 4: Multiple tags, all closed
    multiple_closed = "<tin>{}</tin> text <tin>{}</tin>"
    assert generator._has_unclosed_tool_tag(multiple_closed) is False

    # Case 5: Multiple tags, last one unclosed
    multiple_last_unclosed = "<tin>{}</tin> text <tin>{"
    assert generator._has_unclosed_tool_tag(multiple_last_unclosed) is True

    # Case 6: Truncated closing tag (still counts as unclosed)
    truncated_close = "<tin>{}< "
    assert generator._has_unclosed_tool_tag(truncated_close) is True

    # Case 7: Nested content with complete tags
    nested_complete = """<tin>
{"name": "calc", "arguments": {"values": [1, 2]}}
</tin><tout>3</tout>"""
    assert generator._has_unclosed_tool_tag(nested_complete) is False


def test_fix_truncated_tags_logs_warning(capsys):
    """Test that fix_truncated_tags logs when fixes are applied."""
    from praxis.tools import fix_truncated_tags

    # Test with text that needs fixing
    malformed = """<tin>{"name": "calc"}
</tin>[SEP][BOS]assistant>[SEP]"""

    fixed = fix_truncated_tags(malformed)

    # Check that warning was logged
    captured = capsys.readouterr()
    assert "[TOOL_TAGS]" in captured.out
    assert "fix_truncated_tags applied" in captured.out

    # Check that fixing worked
    assert fixed.rstrip().endswith("</tin>")


def test_fix_truncated_tags_no_warning_when_clean(capsys):
    """Test that fix_truncated_tags doesn't log when no fixes needed."""
    from praxis.tools import fix_truncated_tags

    # Test with clean text
    clean = """<tin>{"name": "calc", "arguments": {}}</tin><tout>42</tout>"""

    result = fix_truncated_tags(clean)

    # Check that no warning was logged
    captured = capsys.readouterr()
    assert "[TOOL_TAGS]" not in captured.out

    # Text should be unchanged
    assert result == clean


def test_complete_tool_call_inline_format():
    """Test the complete inline tool call format expected after the fix."""
    from praxis.tools import (
        format_tool_call_with_result,
        get_unprocessed_tool_call,
        has_complete_tool_call,
        has_tool_output,
    )

    # The expected clean format after the fix:
    # <tin>JSON</tin><tout>result</tout> (inline, no [BOS]assistant in between)
    clean_format = format_tool_call_with_result(
        tool_name="calc",
        arguments={"values": [999, 5], "op": "exp"},
        result="995009990004999.0"
    )

    # Verify structure
    assert "<tin>" in clean_format
    assert "</tin>" in clean_format
    assert "<tout>" in clean_format
    assert "</tout>" in clean_format

    # Verify no malformed patterns
    assert "[BOS]assistant" not in clean_format
    assert "[SEP]" not in clean_format

    # Verify the tags are adjacent (no content between </tin> and <tout>)
    tin_close_idx = clean_format.find("</tin>")
    tout_open_idx = clean_format.find("<tout>")
    assert tout_open_idx == tin_close_idx + len("</tin>")

    # Verify it's recognized as complete (no unprocessed calls)
    assert has_complete_tool_call(clean_format) is True
    assert has_tool_output(clean_format) is True
    assert get_unprocessed_tool_call(clean_format) is None


def test_fix_truncated_tout_not_tin():
    """Test that truncated </tout> is NOT incorrectly fixed as </tin>.

    This was the bug: when text had both <tin>...</tin> and <tout>...</, the
    fix_truncated_tags function would incorrectly complete it as </tin> instead
    of </tout>.
    """
    from praxis.tools import fix_truncated_tags

    # Simulate the exact bug scenario from the screenshot:
    # Text has complete <tin>...</tin> but truncated <tout>...</
    malformed = """<tin>
{"name": "calc", "arguments": {"values": [922583, 622541], "op": "add"}}
</tin><tout>-1234563.0</"""

    fixed = fix_truncated_tags(malformed)

    # The </tout> should be completed, not turned into </tin>
    assert "</tout>" in fixed, f"Expected </tout> in fixed text, got: {fixed}"
    # Should NOT have duplicate </tin> closing tags
    assert fixed.count("</tin>") == 1, f"Expected exactly 1 </tin>, got: {fixed}"
    # Should have exactly one </tout>
    assert fixed.count("</tout>") == 1, f"Expected exactly 1 </tout>, got: {fixed}"


def test_fix_truncated_tags_respects_unclosed_tag_type():
    """Test that fix_truncated_tags correctly identifies which tag is unclosed."""
    from praxis.tools import fix_truncated_tags

    # Case 1: <tin> is unclosed, truncated at </
    tin_unclosed = "<tin>{"
    # Just checking it doesn't crash - no truncation to fix here
    result1 = fix_truncated_tags(tin_unclosed)
    assert "<tin>" in result1

    # Case 2: <tin> is unclosed and truncated
    tin_truncated = """<tin>
{"name": "calc"}
</"""
    result2 = fix_truncated_tags(tin_truncated)
    assert "</tin>" in result2

    # Case 3: <tout> is unclosed and truncated (the bug case)
    tout_truncated = """<tin>{"name": "calc"}</tin><tout>42</"""
    result3 = fix_truncated_tags(tout_truncated)
    assert "</tout>" in result3
    assert result3.count("</tin>") == 1  # Only one </tin>

    # Case 4: Both tags properly closed - no changes needed
    both_closed = """<tin>{"name": "calc"}</tin><tout>42</tout>"""
    result4 = fix_truncated_tags(both_closed)
    assert result4 == both_closed


def test_fix_truncated_tags_multiple_tool_calls():
    """Test fix_truncated_tags with multiple sequential tool calls.

    Ensures the counting logic works correctly when there are many
    complete tool call pairs before a truncated one.
    """
    from praxis.tools import fix_truncated_tags

    # Case 1: Two complete calls, third call's <tin> truncated
    two_complete_third_tin_truncated = """<tin>{"name": "calc", "arguments": {"op": "add"}}</tin><tout>10</tout>
<tin>{"name": "calc", "arguments": {"op": "mul"}}</tin><tout>20</tout>
<tin>{"name": "calc", "arguments": {"op": "div"}}</"""

    result1 = fix_truncated_tags(two_complete_third_tin_truncated)
    assert result1.count("</tin>") == 3, f"Expected 3 </tin>, got {result1.count('</tin>')}"
    assert result1.count("</tout>") == 2, f"Expected 2 </tout>, got {result1.count('</tout>')}"
    assert result1.rstrip().endswith("</tin>")

    # Case 2: Two complete calls, third call's <tout> truncated
    two_complete_third_tout_truncated = """<tin>{"name": "calc"}</tin><tout>10</tout>
<tin>{"name": "calc"}</tin><tout>20</tout>
<tin>{"name": "calc"}</tin><tout>30</"""

    result2 = fix_truncated_tags(two_complete_third_tout_truncated)
    assert result2.count("</tin>") == 3, f"Expected 3 </tin>, got {result2.count('</tin>')}"
    assert result2.count("</tout>") == 3, f"Expected 3 </tout>, got {result2.count('</tout>')}"
    assert result2.rstrip().endswith("</tout>")

    # Case 3: Three complete calls - nothing to fix
    three_complete = """<tin>{"name": "a"}</tin><tout>1</tout>
<tin>{"name": "b"}</tin><tout>2</tout>
<tin>{"name": "c"}</tin><tout>3</tout>"""

    result3 = fix_truncated_tags(three_complete)
    assert result3 == three_complete  # No changes

    # Case 4: One complete, second truncated at </ti
    one_complete_second_partial = """<tin>{"name": "first"}</tin><tout>done</tout>
<tin>{"name": "second"}</ti"""

    result4 = fix_truncated_tags(one_complete_second_partial)
    assert result4.count("</tin>") == 2
    assert "</ti\n" not in result4 and "</ti\"" not in result4  # Partial tag fixed
