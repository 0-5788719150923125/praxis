"""Fixtures shared across the whole test tree.

Importable stubs live in tests/stubs.py; fixtures used by one namespace only
live in that namespace's own conftest.py.
"""

import logging
import sys
import warnings

import pytest

from praxis import PraxisConfig
from praxis.interface.dashboard import TerminalDashboard
from tests.stubs import _Tty, tokenizer_for


@pytest.fixture
def config():
    """A default PraxisConfig."""
    return PraxisConfig()


@pytest.fixture(scope="module")
def prose_tokenizer():
    """The byte-level tokenizer under the text-boundary ``prose`` chat format."""
    return tokenizer_for("prose")


@pytest.fixture(scope="module")
def default_tokenizer():
    """The byte-level tokenizer under the ``default`` chat format."""
    return tokenizer_for("default")


@pytest.fixture
def dashboard(monkeypatch):
    """A dashboard wired to fake streams, never touching the real terminal.

    Constructing one is not side-effect free: it strips the handlers off every
    existing logger, installs its own, forces the root level to INFO and swaps
    the global logger class. All of that has to be put back, or later tests -
    and the interpreter's own atexit logging - inherit a dashboard that no
    longer has a screen.
    """
    tty = _Tty()
    monkeypatch.setattr(sys, "stdout", tty)
    monkeypatch.setattr(sys, "stderr", tty)

    root = logging.getLogger()
    saved_root = (root.handlers[:], root.level)
    saved_class = logging.getLoggerClass()
    saved = {
        name: (logger.handlers[:], logger.propagate, logger.level)
        for name, logger in list(logging.Logger.manager.loggerDict.items())
        if isinstance(logger, logging.Logger)
    }
    saved_showwarning = warnings.showwarning

    dash = TerminalDashboard(seed=1234)
    dash.tty = tty
    yield dash

    try:
        dash.stop()
    except Exception:
        pass
    logging.setLoggerClass(saved_class)
    warnings.showwarning = saved_showwarning
    root.handlers, root.level = saved_root
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if not isinstance(logger, logging.Logger):
            continue
        if name in saved:
            logger.handlers, logger.propagate, logger.level = saved[name]
        else:  # created while the dashboard owned logging
            logger.handlers, logger.propagate, logger.level = [], True, logging.NOTSET
