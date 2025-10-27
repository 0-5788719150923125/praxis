"""Web interface components for dashboard streaming."""

from .buffer import DashboardFrameBuffer
from .renderer import WebDashboardRenderer
from .streamer import DashboardStreamer, get_active_dashboard, register_socketio

__all__ = [
    "DashboardFrameBuffer",
    "WebDashboardRenderer",
    "DashboardStreamer",
    "register_socketio",
    "get_active_dashboard",
]
