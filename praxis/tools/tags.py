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
    """Find the first tool call that doesn't have a corresponding output.

    This function processes tool calls in order (first to last), ensuring that
    multiple tool calls in a single context are all executed sequentially.

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

    # Iterate through all matches in order and find the FIRST one without an output
    # This ensures we process multiple tool calls sequentially, not just the last one
    output_pattern = rf"^\s*<{TOOL_OUTPUT_TAG}>"

    for match in matches:
        # Look for <tout> immediately after this </tin>
        text_after = text[match.end():]

        # If this tool call doesn't have an output, it's unprocessed
        if not re.match(output_pattern, text_after):
            # Try to parse the JSON to make sure it's valid
            try:
                tool_data = json.loads(match.group(1))
                return (tool_data, match.end())
            except json.JSONDecodeError:
                # This tool call has invalid JSON, skip it and check the next one
                continue

    # All tool calls have outputs, nothing to process
    return None


def _get_unclosed_tag(text: str) -> Optional[str]:
    """Determine which tag type (if any) is unclosed in the text.

    Returns:
        'tin' if <tin> is unclosed, 'tout' if <tout> is unclosed, None otherwise.
        If both are unclosed, returns 'tout' since it would be the most recent.
    """
    tin_open = text.count(f"<{TOOL_INPUT_TAG}>")
    tin_close = text.count(f"</{TOOL_INPUT_TAG}>")
    tout_open = text.count(f"<{TOOL_OUTPUT_TAG}>")
    tout_close = text.count(f"</{TOOL_OUTPUT_TAG}>")

    # Check for unclosed tags - if <tout> is unclosed, it's the most recent
    # (since <tout> comes after </tin> in the sequence)
    if tout_open > tout_close:
        return TOOL_OUTPUT_TAG
    if tin_open > tin_close:
        return TOOL_INPUT_TAG
    return None


def fix_truncated_tags(text: str) -> str:
    """Fix common malformed tag patterns.

    Handles cases like:
    - "</" at the very end (incomplete closing tag start)
    - "</tin" without the closing ">"
    - "</t", "</ti" (partial tag names)
    - Extra whitespace or fragments after tags
    - [SEP] and [BOS] tokens that got generated mid-tag

    Note: With the conditional EOS token fix in the generator, this function
    should rarely need to apply fixes. Warnings are logged when fixes are
    applied to help detect gaps in the generation logic.

    Args:
        text: Text potentially containing malformed tags

    Returns:
        Text with fixed tags

    Example:
        >>> fix_truncated_tags('</tin\\n[SEP]')
        '</tin>\\n[SEP]'
    """
    # Store original for comparison
    original_text = text

    # Determine which tag is unclosed (if any) to know what truncated tags should become
    unclosed_tag = _get_unclosed_tag(text)

    # Handle case where generation stopped at just "</" or partial tag names
    # Look for patterns like "</\n", "</\s+", "</t\n", "</ti\n" followed by separators/noise
    # These indicate the model was trying to close a tag but got interrupted
    # Only fix as </tin> if <tin> is the unclosed tag (not <tout>)
    if f"<{TOOL_INPUT_TAG}>" in text and unclosed_tag == TOOL_INPUT_TAG:
        # Replace "</\s*[SEP]" with "</tin>" (tag got cut off at just the opening)
        text = re.sub(
            r"</\s*(?=\[SEP\]|\[BOS\]|$)",
            f"</{TOOL_INPUT_TAG}>",
            text,
        )
        # Replace "</t\s*[SEP]" with "</tin>"
        text = re.sub(
            r"</t\s*(?=\[SEP\]|\[BOS\]|$)",
            f"</{TOOL_INPUT_TAG}>",
            text,
        )
        # Replace "</ti\s*[SEP]" with "</tin>"
        text = re.sub(
            r"</ti\s*(?=\[SEP\]|\[BOS\]|$)",
            f"</{TOOL_INPUT_TAG}>",
            text,
        )
        # Also handle end of string cases
        if text.rstrip().endswith("</"):
            text = text.rstrip() + f"{TOOL_INPUT_TAG}>"
        elif text.rstrip().endswith("</t"):
            text = text.rstrip()[:-1] + f"{TOOL_INPUT_TAG}>"
        elif text.rstrip().endswith("</ti"):
            text = text.rstrip()[:-2] + f"{TOOL_INPUT_TAG}>"

    # Handle truncated </tout> tags when <tout> is the unclosed tag
    if f"<{TOOL_OUTPUT_TAG}>" in text and unclosed_tag == TOOL_OUTPUT_TAG:
        # Replace "</\s*[SEP]" with "</tout>"
        text = re.sub(
            r"</\s*(?=\[SEP\]|\[BOS\]|$)",
            f"</{TOOL_OUTPUT_TAG}>",
            text,
        )
        # Replace "</t\s*[SEP]" with "</tout>"
        text = re.sub(
            r"</t\s*(?=\[SEP\]|\[BOS\]|$)",
            f"</{TOOL_OUTPUT_TAG}>",
            text,
        )
        # Replace "</to\s*[SEP]" with "</tout>"
        text = re.sub(
            r"</to\s*(?=\[SEP\]|\[BOS\]|$)",
            f"</{TOOL_OUTPUT_TAG}>",
            text,
        )
        # Replace "</tou\s*[SEP]" with "</tout>"
        text = re.sub(
            r"</tou\s*(?=\[SEP\]|\[BOS\]|$)",
            f"</{TOOL_OUTPUT_TAG}>",
            text,
        )
        # Handle end of string cases for tout
        if text.rstrip().endswith("</"):
            text = text.rstrip() + f"{TOOL_OUTPUT_TAG}>"
        elif text.rstrip().endswith("</t"):
            text = text.rstrip()[:-1] + f"{TOOL_OUTPUT_TAG}>"
        elif text.rstrip().endswith("</to"):
            text = text.rstrip()[:-2] + f"{TOOL_OUTPUT_TAG}>"
        elif text.rstrip().endswith("</tou"):
            text = text.rstrip()[:-3] + f"{TOOL_OUTPUT_TAG}>"

    # Fix truncated closing tags for tool input (</tin without >)
    if f"</{TOOL_INPUT_TAG}" in text and f"</{TOOL_INPUT_TAG}>" not in text:
        text = text.replace(f"</{TOOL_INPUT_TAG}", f"</{TOOL_INPUT_TAG}>")

    # After fixing, aggressively clean up any separator/role tokens that appear
    # immediately after the closing tag (these are artifacts from generation stopping mid-tag)
    # This handles patterns like: </tin>[SEP], </tin>\n[SEP], </tin>\n[BOS]assistant, etc.
    text = re.sub(
        rf"(</{TOOL_INPUT_TAG}>)\s*\[SEP\](\s*\[BOS\])?\s*(?:assistant\s*>?\s*)?(\[SEP\])?",
        r"\1",
        text,
    )

    # Also clean up lone fragments like "assistant>" or just ">" after closing tags
    text = re.sub(
        rf"(</{TOOL_INPUT_TAG}>)\s*(?:assistant\s*>?|>\s*)",
        r"\1",
        text,
    )

    # Clean up duplicate partial tag names that might have been in the malformed text
    # Pattern: </tin>tin> or </tin>t> or </tin>ti> (from when the model tried to generate
    # the closing tag but it got split, and we already have the complete tag)
    text = re.sub(
        rf"(</{TOOL_INPUT_TAG}>)(?:tin|ti|t)>",
        r"\1",
        text,
    )

    # Fix truncated closing tags for tool output (</tout without >)
    if f"</{TOOL_OUTPUT_TAG}" in text and f"</{TOOL_OUTPUT_TAG}>" not in text:
        text = text.replace(f"</{TOOL_OUTPUT_TAG}", f"</{TOOL_OUTPUT_TAG}>")

    # Clean up duplicate partial tag names for tool output
    # Pattern: </tout>tout> or </tout>tou> etc.
    text = re.sub(
        rf"(</{TOOL_OUTPUT_TAG}>)(?:tout|tou|to|t)>",
        r"\1",
        text,
    )

    # Clean up separator tokens after tool output tags too
    text = re.sub(
        rf"(</{TOOL_OUTPUT_TAG}>)\s*\[SEP\](\s*\[BOS\])?\s*(?:assistant\s*>?\s*)?(\[SEP\])?",
        r"\1",
        text,
    )

    # Log warning if fixes were applied (helps detect gaps in generation logic)
    if text != original_text:
        # Show a snippet of what changed (last 100 chars of each)
        original_snippet = repr(original_text[-100:]) if len(original_text) > 100 else repr(original_text)
        fixed_snippet = repr(text[-100:]) if len(text) > 100 else repr(text)
        print(f"[TOOL_TAGS] fix_truncated_tags applied: {original_snippet} -> {fixed_snippet}")

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
