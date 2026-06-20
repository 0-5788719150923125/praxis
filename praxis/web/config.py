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

# Content Security Policy. Base directives only; integrations contribute their
# own trusted domains via the csp_sources() hook (see build_csp_policy).
CSP_DIRECTIVES = {
    "default-src": ["'self'", "https:"],
    "script-src": [
        "'self'",
        "'unsafe-inline'",
        "'unsafe-eval'",
        "https://cdnjs.cloudflare.com",
        "https://cdn.jsdelivr.net",
    ],
    "style-src": ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
    "font-src": ["'self'", "https://fonts.gstatic.com"],
    "connect-src": ["'self'", "wss:", "ws:", "https:", "http:"],
}


def build_csp_policy(extra_sources=None):
    """Build the CSP header, merging integration-contributed sources by directive.

    Args:
        extra_sources: Optional mapping of directive -> list of extra sources.
    """
    directives = {name: list(vals) for name, vals in CSP_DIRECTIVES.items()}
    for directive, sources in (extra_sources or {}).items():
        bucket = directives.setdefault(directive, [])
        for src in sources:
            if src not in bucket:
                bucket.append(src)
    return (
        "; ".join(f"{name} {' '.join(vals)}" for name, vals in directives.items()) + ";"
    )


# Base policy with no integrations loaded (kept for callers that need a constant).
CSP_POLICY = build_csp_policy()

# Git HTTP backend configuration
GIT_ALLOWED_SERVICES = ["git-upload-pack", "git-receive-pack"]
GIT_READ_ONLY_SERVICE = "git-upload-pack"
