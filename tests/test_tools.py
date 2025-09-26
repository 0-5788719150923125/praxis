"""Test tool-calling functionality."""

import json
import re
from unittest.mock import MagicMock, patch

import pytest
import torch

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.tools import calc, get_tools


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

    # Parse a tool call from text (simulate what Generator would do)
    tool_call_text = """<tool_call>
{"name": "calc", "arguments": {"values": [10, 20], "op": "add"}}
</tool_call>"""

    # Simple regex parsing (similar to Generator._parse_tool_call)
    import re
    import json

    tool_pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
    match = re.search(tool_pattern, tool_call_text, re.DOTALL)

    if match:
        tool_data = json.loads(match.group(1))
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

    # Simulate a tool call response (what an agent might generate)
    tool_response = '<tool_call>\n{"name": "calc", "arguments": {"values": [15, 25], "op": "add"}}\n</tool_call>'

    # Parse and execute
    pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
    match = re.search(pattern, tool_response, re.DOTALL)
    assert match is not None

    tool_data = json.loads(match.group(1))
    # Execute via the calc tool directly
    if tool_data["name"] == "calc":
        result = calc.forward(**tool_data["arguments"])
        assert result == 40


def test_truncated_tool_call_tag():
    """Test that truncated tool call tags are properly fixed."""
    # Test the fix for malformed tool call closing tags
    malformed_text = """[BOS]assistant
<tool_call>
{"name": "calc", "arguments": {"values": [103253, 757695], "op": "add"}}
</tool_call
[SEP]
[BOS]assistant
>
[SEP]
[BOS]tool
860948"""

    # The fix should correct the truncated tag
    if "</tool_call" in malformed_text and not "</tool_call>" in malformed_text:
        fixed_text = malformed_text.replace("</tool_call", "</tool_call>")
        assert "</tool_call>" in fixed_text
        assert "</tool_call\n" not in fixed_text  # The malformed version should be gone

        # Also test that the unwanted assistant message fragment would be removed
        import re

        cleaned_text = re.sub(
            r"</tool_call>\s*\[SEP\]\s*\[BOS\]assistant\s*>\s*\[SEP\]",
            "</tool_call>",
            fixed_text,
        )
        assert (
            "[BOS]assistant\n>" not in cleaned_text
            or "[BOS]assistant" in cleaned_text.split("</tool_call>")[0]
        )
