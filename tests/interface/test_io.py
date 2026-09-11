"""Stopping the dashboard must leave the terminal alone.

The reported failure: Ctrl+C during a run, and the dashboard's box drawing and
charts render *into* the shell's scrollback, interleaved with the shutdown's
own messages. The cause was an ordering bug, not a rendering one. ``stop()``
flipped a flag and immediately left the alternate screen, while the render
thread was still mid-frame or asleep in its 100ms tick - and that thread writes
through a private handle on the real stdout, bypassing every redirection. The
frame it painted next landed, absolutely positioned, on the restored terminal.
"""

import io
import threading

import pytest

from praxis.interface.io import DashboardOutput, LogCapture


class _Tty(io.StringIO):
    """A stand-in terminal that records everything written to it."""

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.chunks = []

    def write(self, s):
        with self.lock:
            self.chunks.append(s)
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return True

    @property
    def text(self):
        with self.lock:
            return "".join(self.chunks)


# ── the redirection contract ─────────────────────────────────────────────


def test_log_capture_is_not_a_tty(dashboard):
    """Otherwise libraries draw progress bars and move a cursor we own."""
    capture = LogCapture(dashboard)
    assert capture.isatty() is False
    assert capture.writable() is True
    assert capture.write("hello\n") == len("hello\n")
    assert capture.encoding


def test_dashboard_output_gate():
    tty = _Tty()
    out = DashboardOutput(tty)
    out.write("visible")
    out.disable()
    out.write("hidden")
    out.flush()
    assert tty.text == "visible"
