"""Indefinite retry on network failures to preserve training reproducibility.

When connectivity drops, streaming datasets must not silently fall back to
empty batches or zero-padding: that would desynchronise data order across
resumptions and runs. These helpers detect transient network errors and wait
for the connection to return, re-raising anything else unchanged.

Offline mode is the automatic fallback: the first hub failure at load time
latches it (enter_offline_mode), after which datasets resolve from the local
cache only and load errors propagate immediately instead of waiting - callers
skip uncached sources (see get_datamodules). The HF_HUB_OFFLINE /
HF_DATASETS_OFFLINE / PRAXIS_OFFLINE env vars force it from boot.
"""

import os
import socket
import time
from typing import Callable, TypeVar

_OFFLINE = False  # latched by enter_offline_mode() on the first hub failure


def hf_offline() -> bool:
    """True when offline mode is active - latched automatically by the first
    hub failure at load time, or forced via env."""
    return _OFFLINE or any(
        os.environ.get(var, "").strip().lower() not in ("", "0", "false")
        for var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "PRAXIS_OFFLINE")
    )


def enter_offline_mode(reason: str) -> None:
    """Latch the process into offline mode: hub libraries resolve from their
    local caches only, and load failures propagate immediately instead of
    waiting for connectivity. Idempotent."""
    global _OFFLINE
    if hf_offline():
        return
    _OFFLINE = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    # Both libraries read their flags at import time; flip the live values too.
    try:
        import huggingface_hub.constants as hub_constants

        hub_constants.HF_HUB_OFFLINE = True
    except Exception:
        pass
    try:
        import datasets.config as datasets_config

        datasets_config.HF_DATASETS_OFFLINE = True
        if hasattr(datasets_config, "HF_HUB_OFFLINE"):
            datasets_config.HF_HUB_OFFLINE = True
    except Exception:
        pass
    print(
        f"[DATA] Hub unreachable ({reason}); entering OFFLINE mode: datasets "
        "resolve from the local cache, uncached ones are skipped.",
        flush=True,
    )
    # Surface on the web app's notification bell - a reminder to retry later.
    try:
        from praxis.interface.state.live_metrics import LiveMetrics

        LiveMetrics().add_event(
            f"Training in OFFLINE mode: hub unreachable ({reason}). Datasets "
            "resolve from local caches; uncached sources are skipped. "
            "Restart when the hub returns to restore the full mixture.",
            level="warning",
        )
    except Exception:
        pass

T = TypeVar("T")

_INITIAL_BACKOFF_SECONDS = 2.0
_MAX_BACKOFF_SECONDS = 60.0

_NETWORK_MODULE_PREFIXES = (
    "requests.",
    "urllib3.",
    "urllib.",
    "http.",
    "httpx.",
    "httpcore.",
    "huggingface_hub.",
    "aiohttp.",
    "fsspec.",
)

# httpx raises plain builtins.RuntimeError for some transport states (e.g.
# huggingface_hub reusing a closed shared client), so module sniffing misses
# them; match on message.
_NETWORK_ERROR_MESSAGES = (
    "client has been closed",
    "Cannot send a request",
)


def is_network_error(exc: BaseException) -> bool:
    """Return True if exc looks like a transient connectivity failure."""
    if isinstance(
        exc, (ConnectionError, TimeoutError, socket.timeout, socket.gaierror)
    ):
        return True

    if isinstance(exc, RuntimeError) and any(
        msg in str(exc) for msg in _NETWORK_ERROR_MESSAGES
    ):
        return True

    module = type(exc).__module__ or ""
    if any(module.startswith(prefix) for prefix in _NETWORK_MODULE_PREFIXES):
        return True

    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return is_network_error(cause)
    return False


def retry_on_network_error(
    func: Callable[[], T],
    label: str = "network call",
    max_attempts: int = 0,
) -> T:
    """Call func(), retrying on network errors with capped backoff -
    indefinitely by default (mid-stream fetches must wait for connectivity to
    preserve data order), or up to ``max_attempts`` when bounded (load-time
    calls, where the caller has a cache fallback).

    Non-network exceptions propagate immediately. Callers must be idempotent,
    since func() may be invoked many times.
    """
    if hf_offline():
        return func()  # offline: waiting for connectivity would never end
    backoff = _INITIAL_BACKOFF_SECONDS
    attempt = 0
    while True:
        try:
            return func()
        except BaseException as exc:
            if not is_network_error(exc):
                raise
            if max_attempts and attempt + 1 >= max_attempts:
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
