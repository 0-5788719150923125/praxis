"""Frame buffer management for dashboard streaming."""

from .renderer import WebDashboardRenderer


class DashboardFrameBuffer:
    """Manages a buffer of dashboard frames with proper synchronization."""

    def __init__(self, max_frames: int = 10):
        self.frames = []
        self.max_frames = max_frames
        self.renderer = WebDashboardRenderer()

    def add_frame(self, frame: list):
        """Add a new frame to the buffer."""
        if len(self.frames) >= self.max_frames:
            self.frames.pop(0)
        self.frames.append(frame)

    def get_latest_frame(self):
        """Get the most recent frame."""
        return self.frames[-1] if self.frames else None

    def get_latest_rendered(self) -> dict:
        """Get the latest frame rendered for web display."""
        frame = self.get_latest_frame()
        if frame:
            return self.renderer.render_frame_for_web(frame)
        return None

    def clear(self):
        """Clear all frames."""
        self.frames = []
