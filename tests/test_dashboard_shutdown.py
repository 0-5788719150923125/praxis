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
import logging
import sys
import threading
import time
import warnings
from types import SimpleNamespace

import pytest

from praxis.interface.dashboard import TerminalDashboard
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


# ── the ordering contract ────────────────────────────────────────────────


def test_stop_waits_for_the_painter_before_releasing_the_screen(dashboard):
    """The reported bug: a frame painted after the terminal was restored.

    Deterministic, not a race: the painter parks inside render_frame and the
    test calls stop() while that paint is in flight - exactly the window the
    signal handler's cleanup thread used to land in.
    """
    in_render = threading.Event()
    painted_after_restore = []

    real_render = dashboard.differential_renderer.render_frame

    def slow_render(frame, out):
        in_render.set()
        time.sleep(0.3)  # a paint that is already under way when stop() lands
        if dashboard.terminal_manager.terminal_restored and out.enabled:
            painted_after_restore.append(frame)
        real_render(frame, out)

    dashboard.differential_renderer.render_frame = slow_render

    dashboard.start()
    assert in_render.wait(5), "the painter never rendered"
    thread = dashboard._render_thread
    dashboard.stop()
    # stop() is supposed to have joined already; wait anyway so the painter has
    # had its chance to misbehave before we judge it.
    thread.join(5)

    assert painted_after_restore == [], "dashboard painted onto a restored terminal"
    assert not dashboard.running
    assert dashboard.terminal_manager.terminal_restored


def test_stop_joins_the_render_thread(dashboard):
    dashboard.start()
    thread = dashboard._render_thread
    assert thread.is_alive()
    assert thread.daemon, "a wedged painter must not outlive the process"

    dashboard.stop()
    assert not thread.is_alive(), "stop() returned while the painter was still up"


def test_nothing_reaches_the_terminal_after_stop(dashboard):
    """Whatever the painter tries next, the tap is shut."""
    dashboard.start()
    time.sleep(0.2)
    dashboard.stop()

    before = dashboard.tty.text
    # Force the paint path directly, as a wedged thread waking late would.
    dashboard._update_screen(["╔" + "═" * 20 + "╗", "╚" + "═" * 20 + "╝"])
    dashboard.dashboard_output.write("╔══ stray frame ══╗")
    dashboard.dashboard_output.flush()

    assert dashboard.tty.text == before, "output escaped after shutdown"


def test_stop_hands_stdout_back(dashboard):
    dashboard.start()
    assert sys.stdout is dashboard.log_capture
    dashboard.stop()
    assert sys.stdout is dashboard.original_stdout
    assert sys.stderr is dashboard.original_stderr


def test_stop_is_idempotent_and_thread_safe(dashboard):
    """stop() arrives from the signal handler, on_fit_end, __exit__ and atexit."""
    dashboard.start()
    time.sleep(0.1)

    errors = []

    def stopper():
        try:
            dashboard.stop()
            dashboard.__exit__(None, None, None)
        except Exception as exc:  # pragma: no cover - the assertion is the point
            errors.append(exc)

    threads = [threading.Thread(target=stopper) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)

    assert errors == []
    assert all(not t.is_alive() for t in threads)
    # Exactly one exit_fullscreen, however many callers raced.
    exits = dashboard.tty.text.count(dashboard.term.exit_fullscreen or "\x1b[?1049l")
    assert exits <= 1, f"terminal released {exits} times"


def test_a_wedged_painter_cannot_reach_the_terminal(dashboard):
    """The join has a timeout; the tap closing is what makes that safe."""
    released = threading.Event()

    def wedged_render(frame, out):
        released.wait(30)  # never returns within the join timeout

    dashboard.differential_renderer.render_frame = wedged_render
    dashboard.start()
    time.sleep(0.2)

    dashboard.stop()  # must return rather than hang
    assert dashboard.terminal_manager.terminal_restored
    assert not dashboard.dashboard_output.enabled

    before = dashboard.tty.text
    released.set()  # the wedged paint finally completes
    dashboard._render_thread.join(5)
    assert dashboard.tty.text == before, "a wedged painter wrote after shutdown"


# ── the force-exit paths bypass every cleanup hook ───────────────────────


def test_release_terminal_leaves_the_alternate_screen(dashboard):
    """os._exit runs no atexit handler, so these paths restore inline."""
    from praxis.callbacks.lightning.signal_handler import SignalHandlerCallback

    cb = SignalHandlerCallback()
    cb.terminal_interface = SimpleNamespace(dashboard=dashboard)

    dashboard.start()
    time.sleep(0.15)
    assert dashboard.terminal_manager.in_fullscreen

    real_stderr = _Tty()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "__stderr__", real_stderr)
        cb._release_terminal()

    assert "\033[?1049l".encode().decode("unicode_escape") in real_stderr.text
    assert "\033[?25h".encode().decode("unicode_escape") in real_stderr.text
    assert not dashboard.running
    assert not dashboard.dashboard_output.enabled


def test_release_terminal_does_not_emit_an_unmatched_rmcup(dashboard):
    """An rmcup we never matched with smcup jumps the cursor into scrollback."""
    from praxis.callbacks.lightning.signal_handler import SignalHandlerCallback

    cb = SignalHandlerCallback()
    cb.terminal_interface = SimpleNamespace(dashboard=dashboard)  # never started

    real_stderr = _Tty()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "__stderr__", real_stderr)
        cb._release_terminal()

    assert "\033[?1049l".encode().decode("unicode_escape") not in real_stderr.text
    assert "\033[?25h".encode().decode("unicode_escape") in real_stderr.text


# ── logging must survive the handover ────────────────────────────────────


def test_logs_after_shutdown_reach_the_real_stderr(dashboard):
    """The dashboard owns every logger; a buffer nobody renders swallows them."""
    dashboard.start()
    time.sleep(0.1)
    dashboard.stop()

    dashboard.add_log("[SHUTDOWN] stopping background services")
    assert "[SHUTDOWN] stopping background services" in dashboard.tty.text


def test_a_never_started_dashboard_still_buffers_logs(dashboard):
    """The headless renderer behind static/terminal.webp drives a dashboard it
    never start()s, and sets ``terminal_restored`` up front precisely because
    it never took the terminal. Keying the "screen is gone" fallback off that
    flag emptied its LOG panel - two different questions, one flag.
    """
    dashboard.terminal_manager.terminal_restored = True  # what the renderer does

    dashboard.add_log("praxis.optim - INFO - lr warmup complete")
    assert "praxis.optim - INFO - lr warmup complete" in list(dashboard.log_buffer)
    assert dashboard.tty.text == "", "log was written out instead of buffered"


def test_logs_are_buffered_while_the_dashboard_owns_the_screen(dashboard):
    dashboard.start()
    time.sleep(0.1)
    dashboard.add_log("praxis.data - INFO - shard 3/8 streamed")
    assert "praxis.data - INFO - shard 3/8 streamed" in list(dashboard.log_buffer)


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
