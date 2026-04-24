"""Indefinite retry on network failures to preserve training reproducibility.

When connectivity drops, streaming datasets must not silently fall back to
empty batches or zero-padding: that would desynchronise data order across
resumptions and runs. These helpers detect transient network errors and wait
for the connection to return, re-raising anything else unchanged.
"""

import socket
import time
from typing import Callable, TypeVar

T = TypeVar("T")

_INITIAL_BACKOFF_SECONDS = 2.0
_MAX_BACKOFF_SECONDS = 60.0

_NETWORK_MODULE_PREFIXES = (
    "requests.",
    "urllib3.",
    "urllib.",
    "http.",
    "huggingface_hub.",
    "aiohttp.",
    "fsspec.",
)


def is_network_error(exc: BaseException) -> bool:
    """Return True if exc looks like a transient connectivity failure."""
    if isinstance(
        exc, (ConnectionError, TimeoutError, socket.timeout, socket.gaierror)
    ):
        return True

    module = type(exc).__module__ or ""
    if any(module.startswith(prefix) for prefix in _NETWORK_MODULE_PREFIXES):
        return True

    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return is_network_error(cause)
    return False


def retry_on_network_error(func: Callable[[], T], label: str = "network call") -> T:
    """Call func(), retrying indefinitely on network errors with capped backoff.

    Non-network exceptions propagate immediately. Callers must be idempotent,
    since func() may be invoked many times.
    """
    backoff = _INITIAL_BACKOFF_SECONDS
    attempt = 0
    while True:
        try:
            return func()
        except BaseException as exc:
            if not is_network_error(exc):
                raise
            attempt += 1
            print(
                f"[NETWORK] {label} failed (attempt {attempt}): "
                f"{type(exc).__name__}: {exc}. "
                f"Retrying in {backoff:.0f}s (waiting indefinitely for connectivity).",
                flush=True,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
