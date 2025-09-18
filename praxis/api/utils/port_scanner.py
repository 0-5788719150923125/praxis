"""Port scanning utilities."""

import socket
from typing import Optional

from ..config import DEFAULT_PORT


def is_port_in_use(port: int, host: str = "localhost") -> bool:
    """Check if a port is already in use.

    Args:
        port: Port number to check
        host: Host to check (default: localhost)

    Returns:
        True if port is in use, False otherwise
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def find_available_port(start_port: int = DEFAULT_PORT, host: str = "localhost") -> int:
    """Find an available port starting from the given port.

    Args:
        start_port: Port to start searching from
        host: Host to check (default: localhost)

    Returns:
        First available port number
    """
    port = start_port
    while is_port_in_use(port, host):
        port += 1
    return port