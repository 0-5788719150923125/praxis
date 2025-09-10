"""Panel rendering for dashboard sections."""

from .utils import TextUtils


class PanelRenderer:
    """Renders various dashboard panels."""
    
    def __init__(self):
        self.text_utils = TextUtils()

    def draw_info_panel(self, info_dict, width, height):
        """Draw the info panel with key/value pairs."""
        lines = []

        # Process items to handle lists that need multiple lines
        display_items = []
        for key, value in info_dict.items():
            if isinstance(value, list):
                # Convert list to string representation
                val_str = str(value)
                max_key_len = width // 3
                max_val_len = width - max_key_len - 3  # -3 for ": " and padding

                # If the list representation is too long, wrap it intelligently
                if len(val_str) > max_val_len:
                    # Smart wrapping that respects list structure
                    wrapped_parts = self.text_utils.wrap_list_string(val_str, max_val_len)
                    # First line with key
                    display_items.append((key, wrapped_parts[0]))
                    # Continuation lines with empty key
                    for part in wrapped_parts[1:]:
                        display_items.append(("", part))
                else:
                    display_items.append((key, val_str))
            else:
                display_items.append((key, value))

        # Format each display item
        for i in range(height):
            if i < len(display_items):
                key, value = display_items[i]
                # Truncate key to fit
                max_key_len = width // 3
                max_val_len = width - max_key_len - 3  # -3 for ": " and padding

                key_str = str(key)[:max_key_len]
                val_str = str(value)[:max_val_len]

                if key:  # Normal line with key
                    line = f" {key_str}: {val_str}"
                else:  # Continuation line
                    line = f" {' ' * max_key_len}  {val_str}"

                lines.append(line.ljust(width)[:width])
            else:
                lines.append(" " * width)

        return lines

    def draw_simulation_panel(self, automata, width, height):
        """Draw a simulation panel."""
        # Account for each cell being 2 characters wide, use complete height
        if (
            automata is None
            or automata.width != (width - 2) // 2
            or automata.height != height
        ):
            from ..visualization.automata import ForestFireAutomata
            automata = ForestFireAutomata((width - 2) // 2, height)

        # Update the game state
        automata.get_next_generation()

        # Convert to ASCII and pad to full width
        lines = automata.to_ascii()
        # Minimal single-space padding for alignment
        return [" " + line + " " for line in lines], automata