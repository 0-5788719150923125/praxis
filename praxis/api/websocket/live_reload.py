"""Live reload WebSocket functionality."""

from flask_socketio import SocketIO


def setup_live_reload(socketio: SocketIO) -> None:
    """Set up live reload WebSocket handlers.

    Args:
        socketio: SocketIO instance
    """

    @socketio.on("connect", namespace="/live-reload")
    def handle_connect():
        """Handle live-reload client connection."""
        pass  # Silent connection

    @socketio.on("disconnect", namespace="/live-reload")
    def handle_disconnect():
        """Handle live-reload client disconnection."""
        pass  # Silent disconnection