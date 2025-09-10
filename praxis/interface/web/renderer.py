"""Web dashboard renderer for converting terminal frames to web display."""

import re


class WebDashboardRenderer:
    """Renders terminal dashboard frames for web display."""

    def __init__(self, target_width: int = 200, target_height: int = 50):
        # Larger default width to accommodate full dashboard
        self.target_width = target_width
        self.target_height = target_height
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def strip_ansi(self, text: str) -> str:
        """Remove ANSI escape sequences."""
        return self.ansi_escape.sub("", text)

    def calculate_visual_width(self, text: str) -> int:
        """Calculate the visual width of text, accounting for Unicode characters."""
        # Strip ANSI codes first
        clean_text = self.strip_ansi(text)
        # Count visual width (simplified - treats all chars as width 1 or 2)
        width = 0
        for char in clean_text:
            if ord(char) > 0x7F:  # Non-ASCII
                # Simplified: assume wide chars are width 2
                width += 2 if ord(char) >= 0x1100 else 1
            else:
                width += 1
        return width

    def extract_dashboard_dimensions(self, frame: list) -> tuple:
        """Extract the actual dimensions of the dashboard frame."""
        if not frame:
            return 0, 0

        # Find the widest line
        max_width = 0
        for line in frame:
            width = self.calculate_visual_width(line)
            max_width = max(max_width, width)

        return max_width, len(frame)

    def render_frame_for_web(self, frame: list) -> dict:
        """
        Render a dashboard frame for web display.
        Returns a dict with the rendered content and metadata.
        """
        if not frame:
            return {
                "html": '<div class="terminal-empty">No dashboard output</div>',
                "text": [],
                "width": 0,
                "height": 0,
            }

        # Get actual dimensions
        actual_width, actual_height = self.extract_dashboard_dimensions(frame)

        # Clean and process each line
        processed_lines = []
        for line in frame:
            # Strip ANSI codes for web display
            clean_line = self.strip_ansi(line)
            processed_lines.append(clean_line)

        # Check if we need to scale or crop
        scale_factor = 1.0
        if actual_width > self.target_width:
            scale_factor = self.target_width / actual_width

        return {
            "text": processed_lines,
            "width": actual_width,
            "height": actual_height,
            "scale_factor": scale_factor,
            "needs_scaling": scale_factor < 1.0,
        }