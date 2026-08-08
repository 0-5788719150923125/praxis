"""Text rendering utilities."""

import re
import textwrap

import wcwidth


class TextUtils:
    """Utilities for text rendering and manipulation."""

    # Single-column stand-in for anything we cannot render in one cell.
    # Must be plain ASCII: U+FFFD and friends are East Asian *Ambiguous*, so a
    # terminal configured for CJK renders them two columns wide and the whole
    # frame shears by one character.
    SAFE_CHAR = "?"
    TAB_WIDTH = 4

    # Printable ASCII and the box-drawing block are all exactly one column, so
    # a line made only of those needs no work at all. This runs on every line
    # of every frame at 10 fps; the regex scan keeps it in C.
    _CELL_SAFE = re.compile(r"^[\x20-\x7e─-╿]*$")

    def __init__(self):
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def strip_ansi(self, text):
        """Remove ANSI escape sequences from the text."""
        return self.ansi_escape.sub("", text)

    def normalize_cells(self, text):
        """
        Return ``text`` rewritten so every character occupies exactly one
        terminal column.

        This is the invariant the whole dashboard rests on: once it holds,
        a string index *is* a terminal column, which is what lets the
        differential renderer address the screen with absolute cursor moves.
        Anything that would break it is rewritten:

        - width 1  -> kept as-is
        - width 0  -> dropped (combining marks, zero-width joiners): the
          terminal gives them no column, so keeping them would slide every
          later character one cell left of where we think it is
        - width 2  -> ``SAFE_CHAR`` (CJK, emoji)
        - width -1 -> ``SAFE_CHAR`` (control chars), except tab, which is
          expanded to spaces because the terminal expands it too
        """
        if not text:
            return ""
        if self._CELL_SAFE.match(text):
            return text

        result = []
        for char in text:
            if char == "\t":
                result.append(" " * self.TAB_WIDTH)
                continue
            width = wcwidth.wcwidth(char)
            if width == 1:
                result.append(char)
            elif width == 0:
                continue
            else:
                result.append(self.SAFE_CHAR)

        return "".join(result)

    def fit_to_width(self, text, width):
        """Normalize text and force it to exactly ``width`` terminal columns."""
        if width <= 0:
            return ""
        cells = self.normalize_cells(text)
        if len(cells) >= width:
            return cells[:width]
        return cells + " " * (width - len(cells))

    def sanitize_text(self, text):
        """
        Sanitize text by replacing problematic characters with safe alternatives.
        Returns sanitized text with consistent character widths.

        Standard whitespace (\\n, \\r, \\t) is preserved verbatim:
        ``wcwidth`` returns -1 for control chars, but these are valid layout
        characters that downstream code (``wrap_text`` for CLI, the browser
        for web) is expected to handle.
        """
        if not text:
            return ""

        result = []
        for char in text:
            if char in ("\n", "\r", "\t"):
                result.append(char)
                continue
            width = wcwidth.wcwidth(char)
            if width < 0 or width > 1:  # Problematic character detected
                result.append("�")  # Using an emoji as a safe replacement
            else:
                result.append(char)

        return "".join(result)

    def truncate_to_width(self, text, width):
        """Truncate text to fit within a given width, accounting for wide characters."""
        if not text:
            return ""
        return self.normalize_cells(text)[: max(0, width)]

    def visual_ljust(self, string, width):
        """Left-justify a string to a specified width, considering character display width."""
        return self.fit_to_width(string, width)

    def visual_len(self, s):
        """Calculate the visual display width of a string."""
        return len(self.normalize_cells(s))

    def wrap_text(self, text, width):
        """Wrap text to fit within a given width, preserving newlines."""
        wrapped_lines = []
        for line in text.splitlines():
            if line == "":  # Handle explicit empty lines (newlines)
                wrapped_lines.append("")  # Just append an empty line
                continue
            # Wrap the text normally
            wrapped = textwrap.wrap(
                line, width=width, break_long_words=True, replace_whitespace=False
            )
            wrapped_lines.extend(wrapped)
        return wrapped_lines

    def wrap_list_string(self, list_str, max_width):
        """Wrap a list string representation intelligently, breaking on commas and spaces."""
        if len(list_str) <= max_width:
            return [list_str]

        wrapped = []
        current_line = ""

        # Try to break on commas followed by spaces
        i = 0
        while i < len(list_str):
            char = list_str[i]
            current_line += char

            # Check if we've reached the line limit
            if len(current_line) >= max_width:
                # Look for the last comma or space to break on
                break_point = -1

                # First try to find a comma followed by space
                for j in range(len(current_line) - 1, -1, -1):
                    if (
                        j > 0
                        and current_line[j - 1] == ","
                        and j < len(current_line)
                        and current_line[j] == " "
                    ):
                        break_point = j
                        break

                # If no comma+space found, try just comma
                if break_point == -1:
                    for j in range(len(current_line) - 1, -1, -1):
                        if current_line[j] == ",":
                            break_point = j + 1  # Keep comma on current line
                            break

                # If no comma found, try space (but not within quotes)
                if break_point == -1:
                    in_quotes = False
                    for j in range(len(current_line) - 1, -1, -1):
                        if current_line[j] in ['"', "'"]:
                            in_quotes = not in_quotes
                        elif current_line[j] == " " and not in_quotes:
                            break_point = j + 1  # Keep space on current line
                            break

                # If we found a break point, use it
                if break_point > 0:
                    wrapped.append(current_line[:break_point])
                    current_line = current_line[break_point:]
                else:
                    # No good break point found, just break at max width
                    wrapped.append(current_line)
                    current_line = ""

            i += 1

        # Add any remaining content
        if current_line:
            wrapped.append(current_line)

        return wrapped
