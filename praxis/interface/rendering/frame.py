"""Frame building and border management."""

from .utils import TextUtils


class FrameBuilder:
    """Builds and manages dashboard frames."""
    
    def __init__(self):
        self.text_utils = TextUtils()

    def correct_borders(self, frame):
        """Correct borders for all lines in a frame."""
        frame_visual_width = self.text_utils.visual_len(frame[0])
        for i in range(1, len(frame) - 1):
            line = frame[i]
            line_visual_len = self.text_utils.visual_len(line)
            if line_visual_len < frame_visual_width:
                padding_needed = frame_visual_width - line_visual_len
                line += " " * padding_needed
            elif line_visual_len > frame_visual_width:
                line = self.text_utils.truncate_to_width(line, frame_visual_width)
            if not line.startswith("║"):
                line = "║" + line[1:]
            if not line.endswith("║"):
                line = line[:-1] + "║"
            frame[i] = line
        return frame

    def check_border_alignment(self, frame):
        """Check if borders are properly aligned."""
        # Assuming the ERROR section is on the first content line after the top border
        error_line_index = 1  # Adjust if necessary
        line = frame[error_line_index]
        expected_length = self.text_utils.visual_len(frame[0])  # Length of the top border
        line_visual_len = self.text_utils.visual_len(line)
        if line_visual_len != expected_length:
            return False
        if not line.startswith("║") or not line.endswith("║"):
            return False
        return True

    def create_top_border(self, half_width, right_width):
        """Create the top border of the frame."""
        return "╔" + "═" * half_width + "╦" + "═" * right_width + "╗"

    def create_bottom_border(self, width):
        """Create the bottom border of the frame."""
        return "╚" + "═" * (width + 1) + "╝"

    def create_footer_separator(self, half_width, right_width):
        """Create the footer separator."""
        return "╠" + "═" * half_width + "╩" + "═" * right_width + "╣"