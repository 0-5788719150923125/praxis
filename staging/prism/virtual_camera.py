"""
Virtual camera output using pyvirtualcam.
Sends processed video frames to a virtual camera device that can be used
in video conferencing applications (Zoom, Teams, Discord, etc.).
"""

import pyvirtualcam
import numpy as np


class VirtualCamera:
    """Wrapper for pyvirtualcam with error handling and utility methods."""

    def __init__(self, width=1280, height=720, fps=30):
        """
        Initialize virtual camera.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None

    def __enter__(self):
        """Context manager entry - start virtual camera."""
        self.cam = pyvirtualcam.Camera(
            width=self.width,
            height=self.height,
            fps=self.fps,
            fmt=pyvirtualcam.PixelFormat.BGR  # OpenCV native format
        )
        print(f"Virtual camera device: {self.cam.device}")
        print(f"Resolution: {self.width}x{self.height} @ {self.fps} FPS")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close virtual camera."""
        if self.cam is not None:
            self.cam.close()
            self.cam = None

    def send(self, frame):
        """
        Send frame to virtual camera.

        Args:
            frame: BGR image (numpy array) from OpenCV

        Returns:
            bool: True if successful, False otherwise
        """
        if self.cam is None:
            return False

        # Resize frame if dimensions don't match
        if frame.shape[:2] != (self.height, self.width):
            import cv2
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        try:
            self.cam.send(frame)
            return True
        except Exception as e:
            print(f"Error sending frame to virtual camera: {e}")
            return False

    def sleep_until_next_frame(self):
        """
        Sleep until next frame time (for consistent FPS).
        Uses adaptive sleep to maintain target frame rate.
        """
        if self.cam is not None:
            self.cam.sleep_until_next_frame()

    @property
    def device_name(self):
        """Get virtual camera device name."""
        if self.cam is not None:
            return self.cam.device
        return None


class PreviewWindow:
    """Optional preview window for debugging (alternative to virtual camera)."""

    def __init__(self, window_name="Face Mask Preview"):
        """
        Initialize preview window.

        Args:
            window_name: Name of the OpenCV window
        """
        self.window_name = window_name
        self.is_open = False

    def __enter__(self):
        """Create preview window."""
        import cv2
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.is_open = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy preview window."""
        import cv2
        if self.is_open:
            cv2.destroyWindow(self.window_name)
            self.is_open = False

    def show(self, frame):
        """
        Display frame in preview window.

        Args:
            frame: BGR image to display

        Returns:
            int: Key code if key pressed, -1 otherwise
        """
        if not self.is_open:
            return -1

        import cv2
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF

    def is_closed(self):
        """Check if window was closed by user."""
        import cv2
        try:
            # Try to get window property - will fail if window closed
            cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
            return False
        except:
            return True
