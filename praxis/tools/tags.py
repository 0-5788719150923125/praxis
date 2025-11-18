"""Utilities for tool call tag formatting and parsing.

This module provides reusable functions for working with tool input (<tin>) and
tool output (<tout>) tags across training, inference, and testing.
"""

import json
import re
from typing import Any, Dict, Optional, Tuple


# Tag constants - change these to update all tool call formatting
TOOL_INPUT_TAG = "tin"
TOOL_OUTPUT_TAG = "tout"


def format_tool_input(tool_name: str, arguments: Dict[str, Any], indent: int = 2) -> str:
    """Format a tool call as <tin>JSON</tin>.

    Args:
        tool_name: Name of the tool to call
        arguments: Dictionary of arguments to pass to the tool
        indent: JSON indentation level (default: 2)

    Returns:
        Formatted tool input string like: <tin>\n{...}\n</tin>

    Example:
        >>> format_tool_input("calc", {"values": [1, 2], "op": "add"})
        '<tin>\\n{"name": "calc", "arguments": {"values": [1, 2], "op": "add"}}\\n</tin>'
    """
    tool_call_data = {"name": tool_name, "arguments": arguments}
    json_str = json.dumps(tool_call_data, indent=indent)
    return f"<{TOOL_INPUT_TAG}>\n{json_str}\n</{TOOL_INPUT_TAG}>"


def format_tool_output(result: Any) -> str:
    """Format a tool result as <tout>result</tout>.

    Args:
        result: The tool execution result (will be converted to string)

    Returns:
        Formatted tool output string like: <tout>result</tout>

    Example:
        >>> format_tool_output(42)
        '<tout>42</tout>'
    """
    return f"<{TOOL_OUTPUT_TAG}>{result}</{TOOL_OUTPUT_TAG}>"


def format_tool_call_with_result(
    tool_name: str, arguments: Dict[str, Any], result: Any, indent: int = 2
) -> str:
    """Format a complete tool call with both input and output.

    Args:
        tool_name: Name of the tool to call
        arguments: Dictionary of arguments
        result: The tool execution result
        indent: JSON indentation level (default: 2)

    Returns:
        Formatted string like: <tin>\n{...}\n</tin><tout>result</tout>

    Example:
        >>> format_tool_call_with_result("calc", {"values": [1, 2], "op": "add"}, 3)
        '<tin>\\n{"name": "calc", "arguments": {"values": [1, 2], "op": "add"}}\\n</tin><tout>3</tout>'
    """
    tool_input = format_tool_input(tool_name, arguments, indent)
    tool_output = format_tool_output(result)
    return f"{tool_input}{tool_output}"


def get_tool_input_pattern() -> str:
    """Get the regex pattern for matching tool input tags.

    Returns:
        Regex pattern string for matching <tin>...</tin>
    """
    return rf"<{TOOL_INPUT_TAG}>\s*({{.*?}})\s*</{TOOL_INPUT_TAG}>"


def get_tool_output_pattern() -> str:
    """Get the regex pattern for matching tool output tags.

    Returns:
        Regex pattern string for matching <tout>...</tout>
    """
    return rf"<{TOOL_OUTPUT_TAG}>(.*?)</{TOOL_OUTPUT_TAG}>"


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse the LAST complete tool call from text.

    Args:
        text: Text containing tool call tags

    Returns:
        Dictionary with 'name' and 'arguments' keys, or None if no valid tool call found

    Example:
        >>> text = '<tin>\\n{"name": "calc", "arguments": {"values": [1, 2], "op": "add"}}\\n</tin>'
        >>> parse_tool_call(text)
        {'name': 'calc', 'arguments': {'values': [1, 2], 'op': 'add'}}
    """
    pattern = get_tool_input_pattern()
    matches = re.findall(pattern, text, re.DOTALL)

    # Process from last to first, returning the first valid JSON
    for match in reversed(matches):
        try:
            tool_data = json.loads(match)
            return tool_data
        except json.JSONDecodeError:
            continue

    return None


def get_unprocessed_tool_call(text: str) -> Optional[Tuple[Dict[str, Any], int]]:
    """Find the last tool call that doesn't have a corresponding output.

    Args:
        text: Text containing tool call tags

    Returns:
        Tuple of (tool_data, match_end_position) or None if no unprocessed call found

    Example:
        >>> text = '<tin>{"name": "calc", "arguments": {"values": [1, 2], "op": "add"}}</tin>'
        >>> result = get_unprocessed_tool_call(text)
        >>> result[0]
        {'name': 'calc', 'arguments': {'values': [1, 2], 'op': 'add'}}
    """
    pattern = get_tool_input_pattern()
    matches = list(re.finditer(pattern, text, re.DOTALL))

    if not matches:
        return None

    # Check the last tool call - does it have a corresponding output?
    last_match = matches[-1]

    # Look for <tout> immediately after this </tin>
    text_after = text[last_match.end():]
    output_pattern = rf"^\s*<{TOOL_OUTPUT_TAG}>"

    # If there's already an output, this tool call is processed
    if re.match(output_pattern, text_after):
        return None

    # Return the last valid unprocessed tool call
    for match in reversed(matches):
        try:
            tool_data = json.loads(match.group(1))
            return (tool_data, match.end())
        except json.JSONDecodeError:
            continue

    return None


def fix_truncated_tags(text: str) -> str:
    """Fix common malformed tag patterns.

    Handles cases like:
    - "</tin" without the closing ">"
    - Extra whitespace or fragments after tags

    Args:
        text: Text potentially containing malformed tags

    Returns:
        Text with fixed tags

    Example:
        >>> fix_truncated_tags('</tin\\n[SEP]')
        '</tin>\\n[SEP]'
    """
    # Fix truncated closing tags for tool input
    if f"</{TOOL_INPUT_TAG}" in text and f"</{TOOL_INPUT_TAG}>" not in text:
        text = text.replace(f"</{TOOL_INPUT_TAG}", f"</{TOOL_INPUT_TAG}>")

        # Remove common fragments after fixed tag
        # Pattern: </tin>[SEP][BOS]assistant>[SEP]
        text = re.sub(
            rf"</{TOOL_INPUT_TAG}>\s*\[SEP\]\s*\[BOS\]assistant\s*>\s*\[SEP\]",
            f"</{TOOL_INPUT_TAG}>",
            text,
        )
        # Also handle cases without special tokens
        text = re.sub(
            rf"</{TOOL_INPUT_TAG}>\s*assistant\s*>\s*",
            f"</{TOOL_INPUT_TAG}>",
            text,
        )

    # Fix truncated closing tags for tool output
    if f"</{TOOL_OUTPUT_TAG}" in text and f"</{TOOL_OUTPUT_TAG}>" not in text:
        text = text.replace(f"</{TOOL_OUTPUT_TAG}", f"</{TOOL_OUTPUT_TAG}>")

    return text


def has_complete_tool_call(text: str) -> bool:
    """Check if text contains at least one complete tool call.

    Args:
        text: Text to check

    Returns:
        True if text contains a complete <tin>...</tin> tag
    """
    pattern = get_tool_input_pattern()
    return bool(re.search(pattern, text, re.DOTALL))


def has_tool_output(text: str) -> bool:
    """Check if text contains at least one tool output.

    Args:
        text: Text to check

    Returns:
        True if text contains a complete <tout>...</tout> tag
    """
    pattern = get_tool_output_pattern()
    return bool(re.search(pattern, text, re.DOTALL))
