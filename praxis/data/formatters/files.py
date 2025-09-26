"""File content formatting utilities for training data."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List


def format_file_as_messages(
    file_path: str,
    content: str,
) -> List[Dict[str, Any]]:
    """
    Format file content as tool-calling messages for training.

    Args:
        file_path: Path to the file being formatted
        content: The file content to format

    Returns:
        List of message dictionaries with tool calls and responses
    """

    # Ensure file_path is a string
    if isinstance(file_path, Path):
        file_path = str(file_path)

    sample_path = file_path
    if random.random() > 0.5:
        # Get clean path
        file_path_obj = Path(file_path)

        # Try to make path relative to cwd for cleaner display
        try:
            sample_path = file_path_obj.relative_to(Path.cwd()).as_posix()
        except (ValueError, RuntimeError):
            sample_path = file_path_obj.as_posix()

    # Ensure sample_path is a string, not a Path object
    if isinstance(sample_path, Path):
        sample_path = str(sample_path)

    # Create tool-calling format
    messages = []

    # Assistant calls the read_file tool
    tool_call = {
        "function": {"name": "read_file", "arguments": {"file_path": sample_path}}
    }

    # Assistant message with tool call (no content needed, just the tool call)
    messages.append({"role": "assistant", "tool_calls": [tool_call]})

    # Tool response with the file content
    messages.append({"role": "tool", "content": content})

    return messages
