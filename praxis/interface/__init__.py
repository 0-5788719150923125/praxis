"""Terminal dashboard interface for Praxis."""

from .dashboard import TerminalDashboard
from .web.streamer import get_active_dashboard, register_socketio

__all__ = ["TerminalDashboard", "register_socketio", "get_active_dashboard"]
