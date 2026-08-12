"""Quiet third-party loggers that flood the CLI with INFO during startup.

HF dataset/hub downloads (httpx / urllib3), file locking, and torch.compile all
log verbosely at INFO, burying our own output. This centralizes the list so the
plain logging setup and the dashboard's logger retrofit agree: the dashboard
forces every logger to INFO, so it must consult this list to keep these quiet.
"""

import logging

# Prefix match: an entry mutes that logger and all of its children.
NOISY_LOGGER_PREFIXES = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "datasets",
    "urllib3",
    "filelock",
    "fsspec",
    "torch._dynamo",
    "torch._inductor",
)


def is_noisy(name: str) -> bool:
    """True if ``name`` is one of the noisy loggers (or a child of one)."""
    return bool(name) and name.startswith(NOISY_LOGGER_PREFIXES)


def quiet_noisy_loggers(level: int = logging.WARNING) -> None:
    """Raise the noisy loggers' threshold so their INFO chatter is dropped."""
    for prefix in NOISY_LOGGER_PREFIXES:
        logging.getLogger(prefix).setLevel(level)
