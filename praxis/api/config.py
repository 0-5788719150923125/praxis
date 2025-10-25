"""Configuration constants for the API module."""

import logging

# Logging configuration
DEFAULT_LOG_LEVEL = logging.WARNING
DEV_LOG_LEVEL = logging.INFO

# Server configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 2100
PORT_RANGE_START = 2100
PORT_RANGE_END = 2120

# Timeout configuration
SERVER_START_TIMEOUT = 5  # seconds
GENERATION_TIMEOUT = 30 * 60  # 30 minutes
GIT_COMMAND_TIMEOUT = 3  # seconds
PORT_CHECK_TIMEOUT = 0.5  # seconds

# WebSocket configuration
SOCKETIO_ASYNC_MODE = "threading"

# File watching configuration
TEMPLATE_CHECK_INTERVAL = 1  # seconds

# CORS configuration
CORS_ORIGINS = "*"
CORS_METHODS = ["GET", "POST", "OPTIONS", "HEAD"]
CORS_HEADERS = ["Content-Type"]

# Content Security Policy
CSP_POLICY = (
    "default-src 'self' https:; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net https://*.ngrok-free.app https://*.ngrok.io; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "connect-src 'self' wss: ws: https: http:;"
)

# Git HTTP backend configuration
GIT_ALLOWED_SERVICES = ["git-upload-pack", "git-receive-pack"]
GIT_READ_ONLY_SERVICE = "git-upload-pack"
