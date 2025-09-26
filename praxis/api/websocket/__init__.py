"""WebSocket functionality for the API."""

from .live_reload import setup_live_reload
from .terminal import setup_terminal_namespace

__all__ = ["setup_live_reload", "setup_terminal_namespace"]
