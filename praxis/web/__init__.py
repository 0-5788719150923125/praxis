"""Praxis web module.

This module provides a clean, modular web server implementation
for both the API and the dashboard frontend.
"""

from .app import api_logger, app, socketio, werkzeug_logger
from .middleware import (
    register_request_middleware,
    register_response_header,
    register_response_middleware,
    register_wsgi_middleware,
)
from .server import APIServer
from .services import Services, start_services

# Export the main interface
__all__ = [
    "APIServer",
    "Services",
    "start_services",
    "app",
    "socketio",
    "api_logger",
    "werkzeug_logger",
    "register_wsgi_middleware",
    "register_request_middleware",
    "register_response_middleware",
    "register_response_header",
]
