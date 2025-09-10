"""Differential rendering for terminal dashboard."""

from typing import List, Optional


class TerminalDifferentialRenderer:
    """
    Character-level differential rendering for terminal output.
    Minimizes terminal update commands by only changing what's necessary.
    """
    
    def __init__(self, term):
        self.term = term
        self.previous_frame: Optional[List[str]] = None
        
    def render_frame(self, new_frame: List[str], dashboard_output) -> None:
        """
        Render a frame with minimal terminal updates.
        Only updates characters that have changed.
        """
        if not new_frame:
            return
            
        # First frame or size changed - full redraw
        if (self.previous_frame is None or 
            len(self.previous_frame) != len(new_frame)):
            self._full_redraw(new_frame, dashboard_output)
            self.previous_frame = new_frame.copy()
            return
        
        # Differential update - only changed characters
        updates = []
        for row, (old_line, new_line) in enumerate(zip(self.previous_frame, new_frame)):
            if old_line != new_line:
                # Find contiguous blocks of changes for efficiency
                changes = self._find_line_changes(old_line, new_line)
                for col_start, col_end, text in changes:
                    updates.append((row, col_start, text))
        
        # Apply updates in batches for efficiency
        self._apply_updates(updates, dashboard_output)
        self.previous_frame = new_frame.copy()
    
    def _full_redraw(self, frame: List[str], dashboard_output) -> None:
        """Perform a full screen redraw."""
        print(
            self.term.home + self.term.clear + self.term.white + "\n".join(frame),
            end="",
            file=dashboard_output,
        )
        dashboard_output.flush()
    
    def _find_line_changes(self, old_line: str, new_line: str) -> List[tuple]:
        """
        Find contiguous blocks of changes in a line.
        Returns list of (start_col, end_col, new_text) tuples.
        """
        if old_line == new_line:
            return []
            
        changes = []
        max_len = max(len(old_line), len(new_line))
        
        # Pad lines for comparison
        old_padded = old_line.ljust(max_len)
        new_padded = new_line.ljust(max_len)
        
        # Find change blocks
        in_change = False
        change_start = 0
        
        for i in range(max_len):
            old_char = old_padded[i] if i < len(old_padded) else ' '
            new_char = new_padded[i] if i < len(new_padded) else ' '
            
            if old_char != new_char:
                if not in_change:
                    change_start = i
                    in_change = True
            else:
                if in_change:
                    # End of change block
                    changes.append((
                        change_start,
                        i,
                        new_padded[change_start:i]
                    ))
                    in_change = False
        
        # Handle change extending to end
        if in_change:
            changes.append((
                change_start,
                max_len,
                new_padded[change_start:].rstrip()
            ))
        
        return changes
    
    def _apply_updates(self, updates: List[tuple], dashboard_output) -> None:
        """Apply character updates to terminal."""
        if not updates:
            return
            
        # Group updates by row to minimize cursor movements
        updates_by_row = {}
        for row, col, text in updates:
            if row not in updates_by_row:
                updates_by_row[row] = []
            updates_by_row[row].append((col, text))
        
        # Sort rows
        sorted_rows = sorted(updates_by_row.keys())
        
        output = []
        for row in sorted_rows:
            # Sort columns within each row
            row_updates = sorted(updates_by_row[row], key=lambda x: x[0])
            
            # Try to minimize cursor movements by combining adjacent updates
            combined = []
            for col, text in row_updates:
                if combined and combined[-1][0] + len(combined[-1][1]) == col:
                    # Adjacent update, combine them
                    combined[-1] = (combined[-1][0], combined[-1][1] + text)
                else:
                    combined.append((col, text))
            
            # Apply combined updates for this row
            for col, text in combined:
                output.append(self.term.move(row, col) + self.term.white + text)
        
        # Send all updates at once
        print("".join(output), end="", file=dashboard_output)
        dashboard_output.flush()
    
    def reset(self) -> None:
        """Reset the renderer state."""
        self.previous_frame = None