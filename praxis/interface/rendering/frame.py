"""Frame building and border management."""

from .utils import TextUtils


class FrameBuilder:
    """Builds and manages dashboard frames."""

    # Characters allowed to sit on a frame edge - the plain rail plus the
    # junctions the footer separator uses.
    LEFT_EDGES = frozenset("\u2551\u2560\u255f\u255e")
    RIGHT_EDGES = frozenset("\u2551\u2563\u2562\u2561")

    def __init__(self):
        self.text_utils = TextUtils()

    def correct_borders(self, frame, width=None):
        """
        Normalize a frame so every line is exactly ``width`` columns wide and
        contains only single-column characters.

        This is the one chokepoint where the frame is made safe to draw. After
        it runs, ``len(line)`` equals the line's column count for every line,
        so the differential renderer's index-to-column arithmetic is exact.
        Panels above are free to be sloppy about odd characters; nothing they
        emit can shear the layout.
        """
        if not frame:
            return frame

        # Prefer the caller's width (the real terminal). Falling back to the
        # top border would let a malformed frame[0] set the width for every
        # other line, which is self-consistent but the wrong size.
        if width is None:
            width = self.text_utils.visual_len(frame[0])
        if width <= 2:
            return frame

        last = len(frame) - 1
        for i, line in enumerate(frame):
            line = self.text_utils.fit_to_width(line, width)
            # Interior lines always carry the vertical rails; content that
            # overran its panel gets clipped rather than pushing them aside.
            # Junction glyphs (the footer's ╠ ╣) are legal edges and kept.
            if 0 < i < last:
                if line[0] not in self.LEFT_EDGES:
                    line = "║" + line[1:]
                if line[-1] not in self.RIGHT_EDGES:
                    line = line[:-1] + "║"
            frame[i] = line
        return frame

    def check_border_alignment(self, frame):
        """Check that every line is the same width and keeps its rails."""
        if not frame:
            return True

        expected = self.text_utils.visual_len(frame[0])
        last = len(frame) - 1
        for i, line in enumerate(frame):
            # Cheap fast path: the invariant makes len() the column count.
            if len(line) != expected and self.text_utils.visual_len(line) != expected:
                return False
            if 0 < i < last and (
                line[0] not in self.LEFT_EDGES or line[-1] not in self.RIGHT_EDGES
            ):
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
