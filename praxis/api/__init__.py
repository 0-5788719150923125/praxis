"""Praxis API module.

This module provides a clean, modular API server implementation.
"""

from .app import api_logger, app, socketio, werkzeug_logger
from .middleware import (
    register_request_middleware,
    register_response_header,
    register_response_middleware,
    register_wsgi_middleware,
)
from .server import APIServer

# Export the main interface
__all__ = [
    "APIServer",
    "app",
    "socketio",
    "api_logger",
    "werkzeug_logger",
    "register_wsgi_middleware",
    "register_request_middleware",
    "register_response_middleware",
    "register_response_header",
]
