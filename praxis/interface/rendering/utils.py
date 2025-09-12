"""Text rendering utilities."""

import re
import textwrap
import wcwidth


class TextUtils:
    """Utilities for text rendering and manipulation."""

    def __init__(self):
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def strip_ansi(self, text):
        """Remove ANSI escape sequences from the text."""
        return self.ansi_escape.sub("", text)

    def sanitize_text(self, text):
        """
        Sanitize text by replacing problematic characters with safe alternatives.
        Returns sanitized text with consistent character widths.
        """
        if not text:
            return ""

        result = []
        for char in text:
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

        # Sanitize the input text first
        sanitized_text = self.sanitize_text(text)

        current_width = 0
        result = []
        for char in sanitized_text:
            char_width = wcwidth.wcwidth(char)
            if char_width < 0:
                char_width = 1  # Treat any remaining problematic characters as width 1
            if current_width + char_width > width:
                break
            result.append(char)
            current_width += char_width

        return "".join(result)

    def visual_ljust(self, string, width):
        """Left-justify a string to a specified width, considering character display width."""
        if not string:
            return " " * width

        # Sanitize the input string first
        sanitized_string = self.sanitize_text(string)

        visual_width = sum(max(wcwidth.wcwidth(char), 0) for char in sanitized_string)
        padding = max(0, width - visual_width)
        return sanitized_string + " " * padding

    def visual_len(self, s):
        """Calculate the visual display width of a string."""
        return sum(max(wcwidth.wcwidth(char), 0) for char in s)

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
