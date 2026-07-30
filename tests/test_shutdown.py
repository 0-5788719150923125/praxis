"""Cancelling a run must look like a cancellation, not a crash.

Three things went wrong when a run was stopped cleanly:

1. The signal handler printed unguarded. Python delivers a handler on the MAIN
   thread between bytecodes, so a ValueError from a closed stdout surfaced as an
   exception rooted in whatever the main thread was executing - a traceback
   through an optimizer scan that had nothing to do with the cause.
2. That exception reached the generic ``except Exception`` in ``run_training``
   and was reported as "fatal error" with a full traceback.
3. The cleanup thread nulled the terminal's dashboard while training was still
   stepping, so the inference hook's print fallback dumped rolling-context text
   onto the terminal the dashboard had just released.
"""

import io
import sys
from types import SimpleNamespace

import pytest

from praxis.callbacks.lightning.signal_handler import SignalHandlerCallback
from praxis.callbacks.lightning.terminal import TerminalInterface


class _DeadStream(io.StringIO):
    """A stream that raises exactly as a closed stdout does."""

    def write(self, s):
        raise ValueError("I/O operation on closed file.")

    def flush(self):
        raise ValueError("I/O operation on closed file.")

    @property
    def closed(self):
        return True


@pytest.fixture
def handler(monkeypatch):
    cb = SignalHandlerCallback()
    cb.trainer_ref = SimpleNamespace(should_stop=False)
    # The cleanup thread is not what these tests are about, and it would race
    # the assertions.
    monkeypatch.setattr(cb, "_deferred_cleanup", lambda: None)
    yield cb
    cb.cuda_manager._shutdown_requested = False


# ── the handler must be total ────────────────────────────────────────────


def test_handler_survives_a_closed_stdout(handler, monkeypatch):
    """The reported failure: ValueError escaping into unrelated main-thread
    code, which then gets classified as a crash."""
    monkeypatch.setattr(sys, "stdout", _DeadStream())
    monkeypatch.setattr(sys, "__stderr__", _DeadStream())

    handler._handle_signal(2, None)  # must not raise

    # And the shutdown still happened - the announcement is not load-bearing.
    assert handler.trainer_ref.should_stop is True
    assert handler.cuda_manager.is_shutting_down() is True


def test_handler_still_flags_shutdown_when_the_trainer_is_gone(handler):
    """Each step is independently guarded, so one failure cannot skip the
    others. The flag is what teardown reads to tell cancel from crash."""

    class Exploding:
        @property
        def should_stop(self):
            raise RuntimeError("trainer already torn down")

        @should_stop.setter
        def should_stop(self, value):
            raise RuntimeError("trainer already torn down")

    handler.trainer_ref = Exploding()
    handler._handle_signal(2, None)
    assert handler.cuda_manager.is_shutting_down() is True


def test_message_reaches_a_live_stdout(handler, monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    handler._handle_signal(2, None)
    assert "Gracefully stopping training" in buf.getvalue()


def test_message_falls_back_to_real_stderr(handler, monkeypatch):
    """stdout is captured into the dashboard's log panel; when it is dead the
    process's own stderr is the next best surface."""
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", _DeadStream())
    monkeypatch.setattr(sys, "__stderr__", err)
    handler._handle_signal(2, None)
    assert "Gracefully stopping training" in err.getvalue()


def test_cleanup_runs_before_the_fit_starts(monkeypatch):
    """A signal during dataset setup arrives before on_fit_start binds the
    terminal interface. Reading it raised AttributeError inside the cleanup
    thread, skipping the dataloader, CUDA and wandb steps AND the force-exit
    watchdog - the last defence against a hung shutdown."""
    cb = SignalHandlerCallback()
    assert cb.terminal_interface is None  # bound in __init__, not on_fit_start

    started = []
    monkeypatch.setattr(
        "praxis.callbacks.lightning.signal_handler.threading.Thread",
        lambda *a, **k: SimpleNamespace(start=lambda: started.append(k.get("target"))),
    )
    cb._deferred_cleanup()  # must not raise
    assert started, "the force-exit watchdog must still be armed"


# ── no internal text on the way out ──────────────────────────────────────


class _Interface(TerminalInterface):
    """Only the inference-display branch is under test."""

    def __init__(self, use_dashboard, dashboard):
        self.use_dashboard = use_dashboard
        self.dashboard = dashboard
        self.headless = False
        self.printed = []

    def print(self, text):
        self.printed.append(text)


def test_torn_down_dashboard_does_not_print_the_context():
    """Cleanup nulls the dashboard while training still steps; the fallback
    print would put the rolling context on the restored terminal."""
    iface = _Interface(use_dashboard=True, dashboard=None)
    iface._show_context("rolling context text")
    assert iface.printed == []


def test_run_without_a_dashboard_still_prints():
    """The fallback is not removed - a run that never had a dashboard is the
    case it exists for."""
    iface = _Interface(use_dashboard=False, dashboard=None)
    iface._show_context("rolling context text")
    assert iface.printed == ["rolling context text"]


def test_live_dashboard_gets_the_text_and_nothing_is_printed():
    class _Dash:
        def __init__(self):
            self.status = None

        def update_status(self, text):
            self.status = text

        def force_redraw(self):
            pass

    dash = _Dash()
    iface = _Interface(use_dashboard=True, dashboard=dash)
    iface._show_context("rolling context text")
    assert dash.status == "rolling context text"
    assert iface.printed == []


def test_shutdown_silences_even_a_live_dashboard():
    """Once shutdown starts nothing more is emitted anywhere - the dashboard
    may be mid-teardown in the cleanup thread."""
    iface = _Interface(use_dashboard=False, dashboard=None)
    iface.begin_shutdown()
    iface._show_context("rolling context text")
    assert iface.printed == []


def test_begin_shutdown_stops_generation():
    """Belt and braces: the hook bails before generating at all, so nothing
    downstream of it can emit either."""
    calls = []

    class _Gen(TerminalInterface):
        def __init__(self):
            self.generator = object()
            self.interval = 10
            self.last_time = None

        def _is_trigger_passed(self, *a):
            calls.append(a)
            return False

    cb = _Gen()
    lm = SimpleNamespace(
        trainer=SimpleNamespace(accumulate_grad_batches=1, global_step=9999)
    )
    cb._generate_text(lm, batch_idx=0, interval=10)
    assert calls, "sanity: normally it gets as far as the trigger check"

    calls.clear()
    cb.begin_shutdown()
    cb._generate_text(lm, batch_idx=0, interval=10)
    assert calls == []


def test_signal_cleanup_flags_the_interface_before_nulling(monkeypatch):
    """Order matters: flag first, then tear down. Nulling first leaves a
    window where the hook sees no dashboard and falls through to print."""
    order = []

    class _Dash:
        def stop(self):
            order.append("dashboard.stop")

        def __exit__(self, *a):
            order.append("dashboard.exit")

    class _Iface:
        dashboard = _Dash()

        def begin_shutdown(self):
            order.append("begin_shutdown")

    cb = SignalHandlerCallback()
    cb.terminal_interface = _Iface()
    monkeypatch.setattr(
        "praxis.callbacks.lightning.signal_handler.threading.Thread",
        lambda *a, **k: SimpleNamespace(start=lambda: None),
    )
    cb._deferred_cleanup()
    assert order[0] == "begin_shutdown"
    assert "dashboard.stop" in order
