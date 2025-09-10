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
        """Apply character updates to terminal preserving text selection."""
        if not updates:
            return
        
        # Begin synchronized update (DCS sequence) - prevents flicker and preserves selection
        # This is supported by modern terminals (kitty, iTerm2, newer gnome-terminal, etc)
        output = ["\033[?2026h"]  # Begin synchronized update
        
        # Group updates by row for efficiency
        updates_by_row = {}
        for row, col, text in updates:
            if row not in updates_by_row:
                updates_by_row[row] = []
            updates_by_row[row].append((col, text))
        
        # Apply updates using absolute positioning
        for row in sorted(updates_by_row.keys()):
            row_updates = sorted(updates_by_row[row], key=lambda x: x[0])
            
            # Coalesce adjacent updates
            merged = []
            for col, text in row_updates:
                if merged and merged[-1][0] + len(merged[-1][1]) == col:
                    merged[-1] = (merged[-1][0], merged[-1][1] + text)
                else:
                    merged.append((col, text))
            
            # Apply each update using CSI CUP (Cursor Position) 
            # This positions cursor without clearing selection in most terminals
            for col, text in merged:
                # CSI row;col H positions cursor (1-indexed)
                # Then write text directly
                output.append(f"\033[{row+1};{col+1}H{self.term.white}{text}")
        
        # End synchronized update
        output.append("\033[?2026l")  # End synchronized update
        
        # Send all updates atomically
        print("".join(output), end="", file=dashboard_output)
        dashboard_output.flush()
    
    def reset(self) -> None:
        """Reset the renderer state."""
        self.previous_frame = None