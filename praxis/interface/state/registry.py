"""Global dashboard registry."""

import threading
import weakref


class DashboardRegistry:
    """Registry for active dashboard instances."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.dashboards = weakref.WeakValueDictionary()
        self.socketio = None

    def register(self, identifier, dashboard):
        """Register a dashboard instance."""
        self.dashboards[identifier] = dashboard

    def get(self, identifier="main"):
        """Get a dashboard instance."""
        return self.dashboards.get(identifier)

    def set_socketio(self, socketio_instance):
        """Set the global SocketIO instance."""
        self.socketio = socketio_instance
