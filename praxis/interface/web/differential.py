"""Differential rendering system for efficient character-level updates."""

import difflib
from typing import Dict, List, Optional, Tuple


class DifferentialRenderer:
    """
    Implements character-level differential rendering for dashboard updates.
    Only transmits changed characters instead of full frames.
    """

    def __init__(self):
        self.previous_frame: Optional[List[str]] = None
        self.current_buffer: List[List[str]] = []  # 2D character buffer

    def compute_diff(self, new_frame: List[str]) -> Dict:
        """
        Compute the differences between the previous and new frame.
        Returns a dict containing only the changed characters and their positions.
        """
        if not new_frame:
            return {"type": "empty", "changes": []}

        # If no previous frame, send full update
        if self.previous_frame is None:
            self.previous_frame = new_frame.copy()
            self._update_buffer(new_frame)
            return {
                "type": "full",
                "frame": new_frame,
                "width": max(len(line) for line in new_frame) if new_frame else 0,
                "height": len(new_frame),
            }

        changes = []

        # Ensure frames have same number of lines
        max_lines = max(len(self.previous_frame), len(new_frame))

        for row in range(max_lines):
            old_line = (
                self.previous_frame[row] if row < len(self.previous_frame) else ""
            )
            new_line = new_frame[row] if row < len(new_frame) else ""

            if old_line != new_line:
                # Find character-level differences in this line
                line_changes = self._compute_line_diff(row, old_line, new_line)
                if line_changes:
                    changes.extend(line_changes)

        # Update previous frame
        self.previous_frame = new_frame.copy()
        self._update_buffer(new_frame)

        return {
            "type": "diff",
            "changes": changes,
            "width": max(len(line) for line in new_frame) if new_frame else 0,
            "height": len(new_frame),
        }

    def _compute_line_diff(self, row: int, old_line: str, new_line: str) -> List[Dict]:
        """
        Compute character-level differences for a single line.
        Returns a list of change objects with position and new content.
        """
        changes = []

        # Use difflib to find matching blocks
        matcher = difflib.SequenceMatcher(None, old_line, new_line)

        # Get all the non-matching regions
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            elif tag == "replace":
                # Characters were replaced
                changes.append(
                    {
                        "row": row,
                        "col": i1,
                        "text": new_line[j1:j2],
                        "length": i2 - i1,  # Length of text being replaced
                    }
                )
            elif tag == "delete":
                # Characters were deleted
                changes.append({"row": row, "col": i1, "text": "", "length": i2 - i1})
            elif tag == "insert":
                # Characters were inserted
                changes.append(
                    {
                        "row": row,
                        "col": i1,
                        "text": new_line[j1:j2],
                        "length": 0,  # Nothing being replaced, just inserting
                    }
                )

        return changes

    def _update_buffer(self, frame: List[str]):
        """Update the internal 2D character buffer."""
        self.current_buffer = []
        for line in frame:
            self.current_buffer.append(list(line))

    def get_character_at(self, row: int, col: int) -> str:
        """Get a character at a specific position in the current buffer."""
        if row < len(self.current_buffer) and col < len(self.current_buffer[row]):
            return self.current_buffer[row][col]
        return " "

    def reset(self):
        """Reset the renderer state."""
        self.previous_frame = None
        self.current_buffer = []


class OptimizedDifferentialRenderer(DifferentialRenderer):
    """
    Optimized version that batches consecutive character changes.
    Reduces the number of update operations for better performance.
    """

    def _compute_line_diff(self, row: int, old_line: str, new_line: str) -> List[Dict]:
        """
        Compute character-level differences with batching optimization.
        Consecutive changes are merged into single update operations.
        """
        if not old_line and not new_line:
            return []

        # If entire line changed, send as single update
        if len(old_line) == 0:
            return [{"row": row, "col": 0, "text": new_line, "length": 0}]

        if len(new_line) == 0:
            return [{"row": row, "col": 0, "text": "", "length": len(old_line)}]

        changes = []
        max_len = max(len(old_line), len(new_line))

        # Pad lines to same length for comparison
        old_line = old_line.ljust(max_len)
        new_line = new_line.ljust(max_len)

        # Find contiguous blocks of changes
        change_start = None

        for col in range(max_len):
            old_char = old_line[col] if col < len(old_line) else " "
            new_char = new_line[col] if col < len(new_line) else " "

            if old_char != new_char:
                if change_start is None:
                    change_start = col
            else:
                if change_start is not None:
                    # End of change block
                    changes.append(
                        {
                            "row": row,
                            "col": change_start,
                            "text": new_line[change_start:col],
                            "length": col - change_start,
                        }
                    )
                    change_start = None

        # Handle change that extends to end of line
        if change_start is not None:
            changes.append(
                {
                    "row": row,
                    "col": change_start,
                    "text": new_line[change_start:].rstrip(),
                    "length": len(old_line[change_start:].rstrip()),
                }
            )

        return changes
