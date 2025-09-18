"""Middleware registration and management."""

from typing import List, Tuple, Callable, Any
from flask import Request, Response

# Global middleware registries
_wsgi_middleware: List[Callable] = []
_request_middleware: List[Callable] = []
_response_middleware: List[Callable] = []
_response_headers: List[Tuple[str, str]] = []


def register_wsgi_middleware(middleware_func: Callable) -> None:
    """Register a WSGI middleware function from modules."""
    _wsgi_middleware.append(middleware_func)


def register_request_middleware(func: Callable) -> None:
    """Register a request middleware function from modules."""
    _request_middleware.append(func)


def register_response_middleware(func: Callable) -> None:
    """Register a response middleware function from modules."""
    _response_middleware.append(func)


def register_response_header(header_name: str, header_value: str) -> None:
    """Register a header to be added to all responses."""
    _response_headers.append((header_name, header_value))


def get_wsgi_middleware() -> List[Callable]:
    """Get all registered WSGI middleware."""
    return _wsgi_middleware


def get_request_middleware() -> List[Callable]:
    """Get all registered request middleware."""
    return _request_middleware


def get_response_middleware() -> List[Callable]:
    """Get all registered response middleware."""
    return _response_middleware


def get_response_headers() -> List[Tuple[str, str]]:
    """Get all registered response headers."""
    return _response_headers


def apply_wsgi_middleware(app: Any) -> None:
    """Apply all registered WSGI middleware to the app."""
    for middleware in _wsgi_middleware:
        app.wsgi_app = middleware(app.wsgi_app)


def process_request_middleware(request: Request) -> Any:
    """Process all registered request middleware."""
    for middleware in _request_middleware:
        result = middleware(request, None)
        if result is not None:
            return result
    return None


def process_response_middleware(request: Request, response: Response) -> Response:
    """Process all registered response middleware and headers."""
    # Add any registered headers
    for header_name, header_value in _response_headers:
        response.headers[header_name] = header_value

    # Process middleware functions
    for middleware in _response_middleware:
        middleware(request, response)
    return response