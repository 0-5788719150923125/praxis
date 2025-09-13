"""Dashboard streaming functionality for web clients."""

import threading
import time
import weakref

from .buffer import DashboardFrameBuffer
from .differential import OptimizedDifferentialRenderer
from .renderer import WebDashboardRenderer

# Global registry for dashboard streaming
_active_dashboards = weakref.WeakValueDictionary()
_dashboard_lock = threading.Lock()
_global_socketio = None


def register_socketio(socketio_instance):
    """Register the global SocketIO instance for dashboard streaming."""
    global _global_socketio
    _global_socketio = socketio_instance
    # SocketIO registered for dashboard streaming


def get_active_dashboard(identifier="main"):
    """Get the active dashboard instance if available."""
    return _active_dashboards.get(identifier)


class DashboardStreamer:
    """Streams dashboard frames to web clients via SocketIO."""

    def __init__(self, dashboard):
        self.dashboard = weakref.ref(dashboard)
        self.streaming = False
        self.stream_thread = None
        self.last_frame = None
        self.frame_buffer = DashboardFrameBuffer()
        # Use wider target width for better display
        self.renderer = WebDashboardRenderer(target_width=200)
        # Add differential renderer for efficient updates
        self.differential_renderer = OptimizedDifferentialRenderer()

    def start(self):
        """Start streaming dashboard output."""
        if self.streaming:
            return

        print("Dashboard streaming started.")
        self.streaming = True
        self.stream_thread = threading.Thread(target=self._stream_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()

    def stop(self):
        """Stop streaming dashboard output."""
        self.streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
            self.stream_thread = None
        print("Dashboard streaming stopped")

    def get_current_frame(self):
        """Get the current dashboard frame."""
        dashboard = self.dashboard()
        if dashboard and hasattr(dashboard, "previous_frame"):
            return dashboard.previous_frame
        return None

    def get_buffered_frames(self):
        """Get all buffered frames."""
        return list(self.frame_buffer)

    def _stream_loop(self):
        """Main streaming loop."""
        global _global_socketio

        while self.streaming:
            try:
                dashboard = self.dashboard()
                if not dashboard:
                    # Dashboard was garbage collected
                    break

                # Get current frame
                if hasattr(dashboard, "previous_frame"):
                    frame = dashboard.previous_frame

                    # Check if frame changed
                    if frame != self.last_frame and frame is not None:
                        self.last_frame = frame

                        # Add to buffer
                        if hasattr(self.frame_buffer, "add_frame"):
                            self.frame_buffer.add_frame(frame)
                        else:
                            self.frame_buffer.append(frame)

                        # Stream to web clients if socketio is available
                        if _global_socketio:
                            try:
                                # Strip ANSI codes from frame before diff
                                clean_frame = []
                                if self.renderer:
                                    for line in frame:
                                        clean_frame.append(
                                            self.renderer.strip_ansi(line)
                                        )
                                else:
                                    clean_frame = frame

                                # Compute differential update
                                diff_data = self.differential_renderer.compute_diff(
                                    clean_frame
                                )

                                # Send differential update to clients
                                _global_socketio.emit(
                                    "dashboard_update",
                                    {
                                        "type": diff_data["type"],
                                        "changes": diff_data.get("changes", []),
                                        "frame": diff_data.get("frame", None),
                                        "width": diff_data.get("width", 0),
                                        "height": diff_data.get("height", 0),
                                        "timestamp": time.time(),
                                    },
                                    namespace="/terminal",
                                )
                            except Exception as e:
                                print(f"Error emitting frame: {e}")

                # Also capture dashboard state
                if _global_socketio and dashboard:
                    state = {
                        "status": getattr(dashboard, "status_text", "Unknown"),
                        "step": getattr(dashboard, "step", 0),
                        "batch": getattr(dashboard, "batch", 0),
                        "mode": getattr(dashboard, "mode", "unknown"),
                        "running": getattr(dashboard, "running", False),
                    }

                    try:
                        _global_socketio.emit(
                            "dashboard_state", state, namespace="/terminal"
                        )
                    except:
                        pass

                time.sleep(0.1)  # Match dashboard update rate

            except Exception as e:
                print(f"Error in dashboard streaming: {e}")
                time.sleep(1)

        print("Dashboard streaming loop ended")


# Export registry access
def register_dashboard(identifier, dashboard):
    """Register a dashboard instance."""
    with _dashboard_lock:
        _active_dashboards[identifier] = dashboard
        if _global_socketio:
            # Start streaming if socketio is available
            if hasattr(dashboard, "_streamer"):
                dashboard._streamer.start()
