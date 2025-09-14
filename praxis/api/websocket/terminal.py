"""Terminal/dashboard WebSocket streaming."""

import time
from typing import Optional, Any
from flask_socketio import Namespace, emit, SocketIO


def setup_terminal_namespace(socketio: SocketIO, dashboard: Optional[Any] = None) -> None:
    """Set up terminal WebSocket namespace for dashboard streaming.

    Args:
        socketio: SocketIO instance
        dashboard: Optional dashboard instance for streaming
    """
    # Try to import interface module for dashboard streaming
    try:
        from praxis import interface
        terminal_available = True
    except ImportError:
        terminal_available = False
        interface = None

    if not terminal_available:
        return

    # Register socketio for dashboard streaming
    interface.register_socketio(socketio)

    class TerminalNamespace(Namespace):
        """WebSocket namespace for terminal/dashboard interaction."""

        def on_connect(self):
            """Send current dashboard state on connect."""
            dashboard = interface.get_active_dashboard("main")
            if dashboard and hasattr(dashboard, "_streamer"):
                # Get the latest rendered frame
                frame = dashboard._streamer.get_current_frame()
                if frame:
                    # Strip ANSI codes
                    clean_frame = []
                    if dashboard._streamer.renderer:
                        for line in frame:
                            clean_frame.append(
                                dashboard._streamer.renderer.strip_ansi(line)
                            )
                    else:
                        clean_frame = frame

                    # Send as full update in new format
                    emit(
                        "dashboard_update",
                        {
                            "type": "full",
                            "frame": clean_frame,
                            "width": (
                                max(len(line) for line in clean_frame)
                                if clean_frame
                                else 0
                            ),
                            "height": len(clean_frame),
                            "timestamp": time.time(),
                        },
                    )

        def on_disconnect(self):
            """Handle client disconnection."""
            pass

        def on_start_capture(self, data):
            """Connect to existing dashboard."""
            dashboard = interface.get_active_dashboard("main")
            if dashboard and hasattr(dashboard, "_streamer"):
                dashboard._streamer.start()
                emit("capture_started", {"status": "connected_to_existing"})

                # Send current frame if available
                frame = dashboard._streamer.get_current_frame()
                if frame:
                    # Strip ANSI codes
                    clean_frame = []
                    if dashboard._streamer.renderer:
                        for line in frame:
                            clean_frame.append(
                                dashboard._streamer.renderer.strip_ansi(line)
                            )
                    else:
                        clean_frame = frame

                    # Send as full update in new format
                    emit(
                        "dashboard_update",
                        {
                            "type": "full",
                            "frame": clean_frame,
                            "width": (
                                max(len(line) for line in clean_frame)
                                if clean_frame
                                else 0
                            ),
                            "height": len(clean_frame),
                            "timestamp": time.time(),
                        },
                    )
            else:
                emit("capture_started", {"status": "no_dashboard_found"})

        def on_stop_capture(self):
            """Stop dashboard streaming."""
            dashboard = interface.get_active_dashboard("main")
            if dashboard and hasattr(dashboard, "_streamer"):
                dashboard._streamer.stop()
            emit("capture_stopped", {"status": "ok"})

    # Register the terminal namespace
    socketio.on_namespace(TerminalNamespace("/terminal"))