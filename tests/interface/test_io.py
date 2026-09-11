"""Dashboard IO (praxis/interface/io): the stream stand-ins that keep output
off a terminal the dashboard owns, and off one it has released."""

from types import SimpleNamespace

from praxis.interface.io import DashboardOutput, LogCapture
from tests.stubs import _Tty


def test_log_capture_is_not_a_tty():
    """Otherwise libraries draw progress bars and move a cursor we own."""
    logged = []
    capture = LogCapture(SimpleNamespace(add_log=logged.append))
    assert capture.isatty() is False
    assert capture.writable() is True
    assert capture.write("hello\n") == len("hello\n")
    assert logged == ["hello"]
    assert capture.encoding


def test_dashboard_output_gate():
    tty = _Tty()
    out = DashboardOutput(tty)
    out.write("visible")
    out.disable()
    out.write("hidden")
    out.flush()
    assert tty.text == "visible"
