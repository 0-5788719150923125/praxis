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

import contextlib
import os
import socket
import time
from typing import Callable, TypeVar

_OFFLINE = False  # latched by enter_offline_mode() on the first hub failure


@contextlib.contextmanager
def force_offline():
    """Temporarily force hub libraries offline, then restore prior state.

    Unlike enter_offline_mode (a permanent process latch), this is scoped to a
    single load. It flips the *runtime* config flags datasets/huggingface_hub
    actually consult in download_and_prepare - DownloadConfig(local_files_only)
    alone does not reliably stop a builder from downloading - so a cache read
    can never fall through to a full network download. Restores env + flags on
    exit so unrelated datasets stay online.
    """
    saved_env = {
        var: os.environ.get(var)
        for var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    }
    for var in saved_env:
        os.environ[var] = "1"
    hub_constants = datasets_config = None
    saved_hub = saved_ds = saved_ds_hub = None
    try:
        import huggingface_hub.constants as hub_constants

        saved_hub = getattr(hub_constants, "HF_HUB_OFFLINE", None)
        hub_constants.HF_HUB_OFFLINE = True
    except Exception:
        hub_constants = None
    try:
        import datasets.config as datasets_config

        saved_ds = getattr(datasets_config, "HF_DATASETS_OFFLINE", None)
        datasets_config.HF_DATASETS_OFFLINE = True
        if hasattr(datasets_config, "HF_HUB_OFFLINE"):
            saved_ds_hub = datasets_config.HF_HUB_OFFLINE
            datasets_config.HF_HUB_OFFLINE = True
    except Exception:
        datasets_config = None
    try:
        yield
    finally:
        for var, val in saved_env.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val
        if hub_constants is not None and saved_hub is not None:
            hub_constants.HF_HUB_OFFLINE = saved_hub
        if datasets_config is not None:
            if saved_ds is not None:
                datasets_config.HF_DATASETS_OFFLINE = saved_ds
            if saved_ds_hub is not None and hasattr(datasets_config, "HF_HUB_OFFLINE"):
                datasets_config.HF_HUB_OFFLINE = saved_ds_hub


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


def hub_reachable(timeout: float = 5.0, attempts: int = 3, delay: float = 1.0) -> bool:
    """Cheap TCP probe of the hub - distinguishes a dataset-specific failure
    (hub up: skip that dataset) from real connectivity loss (latch offline).

    Retried: this probe gates a *process-wide* offline latch, so a single
    transient DNS/connect blip (the same kind that makes one dataset's load
    fail) must not be enough to declare the hub down and disable every other
    dataset. Only a sustained outage - all attempts failing - latches offline.
    """
    for attempt in range(attempts):
        try:
            with socket.create_connection(("huggingface.co", 443), timeout=timeout):
                return True
        except OSError:
            if attempt + 1 < attempts:
                time.sleep(delay)
    return False


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

# A closed client never heals by waiting - every retry reuses the same dead
# object. The caller must rebuild its stream instead.
_UNRECOVERABLE_MESSAGES = ("client has been closed",)


def is_unrecoverable(exc: BaseException) -> bool:
    """True for transport states that retrying the same callable cannot fix."""
    if any(msg in str(exc) for msg in _UNRECOVERABLE_MESSAGES):
        return True
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return is_unrecoverable(cause)
    return False


def reset_hub_session() -> None:
    """Drop huggingface_hub's shared HTTP client (and cached HF filesystems) so
    the next request builds them fresh.

    The shared httpx client can be left CLOSED but still referenced - a fork's
    at-fork hook, a checkpoint-resume fork, or an SSL reset can close it - and
    reusing a closed client raises ``RuntimeError: ... client has been closed``,
    which then fails every dataset load for the rest of the process. Resetting
    here is what makes a load *retry* actually fresh (the bare retry reused the
    dead client). Best-effort: never raise.
    """
    try:
        from huggingface_hub.utils import _http

        _http.close_session()  # sets the global client to None -> recreated lazily
    except Exception:
        pass
    # datasets streaming resolves through a cached fsspec HfFileSystem that holds
    # its own reference to the (now stale) client; clear the instance cache so it
    # rebuilds with a live one.
    try:
        from huggingface_hub import HfFileSystem

        HfFileSystem.clear_instance_cache()
    except Exception:
        pass


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


# Cache-resolution failures: the dataset isn't usable locally (uncached, a
# partial '.incomplete' download, or a config-hash mismatch in the cache).
# Like a network error, these mean "skip this dataset", not "crash the run".
_MISSING_DATA_MESSAGES = (
    "couldn't find cache",
    "couldn't find file",
    "couldn't reach",
    ".incomplete",
    "offlinemodeisenabled",
    "offline mode is enabled",
)


def is_skippable_load_error(exc: BaseException) -> bool:
    """True when a dataset load failed for a reason that should skip just that
    dataset rather than abort training: a network error, or the data simply not
    being available locally (cache miss / corruption). Genuine programming
    errors (bad config, type errors) still propagate.
    """
    if is_network_error(exc):
        return True
    if isinstance(exc, FileNotFoundError):
        return True
    msg = str(exc).lower()
    if any(s in msg for s in _MISSING_DATA_MESSAGES):
        return True
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return is_skippable_load_error(cause)
    return False


def retry_on_network_error(
    func: Callable[[], T],
    label: str = "network call",
    max_attempts: int = 0,
) -> T:
    """Call func(), retrying on network errors with capped backoff -
    indefinitely by default (mid-stream fetches wait for connectivity to
    preserve data order), or up to ``max_attempts`` when bounded (load-time
    calls, where the caller has a cache fallback). If the hub itself stops
    answering a TCP probe, offline mode latches and the error propagates so
    callers fall back to local caches instead of hanging training.

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
            if is_unrecoverable(exc):
                raise  # dead client object; waiting can't revive it
            if max_attempts and attempt + 1 >= max_attempts:
                raise
            if not hub_reachable():
                # The hub itself is down, not just this fetch. Don't wait on
                # connectivity that may not return - but don't latch the whole
                # process offline either: a blip while loading one dataset must
                # not disable the rest. Raise so THIS caller falls back to its
                # local cache (per-dataset); other datasets retry independently.
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
