"""File content formatting utilities for training data."""

import random
from pathlib import Path
from typing import Dict, List


def format_file_as_messages(
    file_path: str,
    content: str,
) -> List[Dict[str, str]]:
    """
    Format file content as a simple message structure for training.

    Args:
        file_path: Path to the file being formatted
        content: The file content to format

    Returns:
        List of message dictionaries with role and content
    """

    sample_path = file_path
    if random.random() > 0.5:
        # Get clean path
        file_path_obj = Path(file_path)

        # Try to make path relative to cwd for cleaner display
        try:
            sample_path = file_path_obj.relative_to(Path.cwd()).as_posix()
        except (ValueError, RuntimeError):
            sample_path = file_path_obj.as_posix()

    # Simple message structure
    return [{"role": "assistant", "content": f"File: {sample_path}\n\n{content}"}]
