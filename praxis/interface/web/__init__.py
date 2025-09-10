"""Web interface components for dashboard streaming."""

from .buffer import DashboardFrameBuffer
from .renderer import WebDashboardRenderer
from .streamer import DashboardStreamer, register_socketio, get_active_dashboard

__all__ = [
    'DashboardFrameBuffer',
    'WebDashboardRenderer', 
    'DashboardStreamer',
    'register_socketio',
    'get_active_dashboard'
]