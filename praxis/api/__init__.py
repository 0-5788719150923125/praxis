"""Praxis API module.

This module provides a clean, modular API server implementation.
"""

from .server import APIServer
from .app import app, socketio, api_logger, werkzeug_logger
from .middleware import (
    register_wsgi_middleware,
    register_request_middleware,
    register_response_middleware,
    register_response_header,
)

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