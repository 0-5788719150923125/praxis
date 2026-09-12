"""How a run ends, and what it reports for having ended that way.

SIGTERM is the case worth pinning: it is how every container stop arrives
(`docker stop`, `compose down`, systemd), and Lightning signals it by raising
``SIGTERMException``, which subclasses ``SystemExit`` rather than ``Exception``.
It therefore matches neither of the handlers a reader would expect to cover it,
and used to leave through ``main()`` - skipping the whole graceful teardown and
exiting 0, so a killed run was indistinguishable from a finished one.
"""

from types import SimpleNamespace

import pytest

from praxis.trainers import runtime


@pytest.fixture
def teardown(monkeypatch):
    """Capture what run_training asks graceful_shutdown to do."""
    calls = []

    def fake_shutdown(api_server, exit_code=0, reason=""):
        calls.append({"exit_code": exit_code, "reason": reason})

    monkeypatch.setattr("praxis.utils.graceful_shutdown", fake_shutdown)
    # Re-installing signal handlers is real work this test does not need, and
    # pytest runs on the main thread where it would actually take effect.
    monkeypatch.setattr(runtime.signal, "signal", lambda *a, **k: None)
    return calls


def _run(raises=None):
    """Drive run_training against a trainer whose fit does one thing."""

    class Trainer:
        def fit(self, *args, **kwargs):
            if raises is not None:
                raise raises

    services = SimpleNamespace(api_server=None)
    return runtime.run_training(Trainer(), None, None, None, services, None)


def test_a_completed_fit_exits_zero(teardown):
    _run()
    assert teardown == [{"exit_code": 0, "reason": "training complete"}]


def test_sigterm_tears_down_gracefully_and_does_not_report_success(teardown):
    """The bug: `docker events` recorded a terminated run as `exitCode=0`."""
    from lightning.pytorch.utilities.exceptions import SIGTERMException

    code = _run(raises=SIGTERMException())

    assert code == 143, "128 + SIGTERM, the shell convention"
    assert teardown == [{"exit_code": 143, "reason": "terminated"}]


def test_sigterm_is_not_an_exception_subclass():
    """Why the dedicated branch exists at all. If Lightning ever changes this,
    the branch is redundant and this test says so."""
    from lightning.pytorch.utilities.exceptions import SIGTERMException

    assert issubclass(SIGTERMException, SystemExit)
    assert not issubclass(SIGTERMException, Exception)


def test_an_explicit_nonzero_exit_code_is_preserved(teardown):
    """A deliberate `sys.exit(2)` from inside the fit still means 2."""
    assert _run(raises=SystemExit(2)) == 2
    assert teardown[0]["exit_code"] == 2


def test_keyboard_interrupt_still_exits_130(teardown):
    assert _run(raises=KeyboardInterrupt()) == 130
    assert teardown == [{"exit_code": 130, "reason": "interrupted"}]


def test_a_real_crash_still_exits_one(teardown):
    assert _run(raises=RuntimeError("boom")) == 1
    assert teardown[0]["exit_code"] == 1
