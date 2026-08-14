"""Nothing on a background thread may swap the process-global ``sys.stdout``.

``contextlib.redirect_stdout`` mutates a PROCESS-GLOBAL. Used from the Flask API
thread or a build thread, it silently redirects every other thread's output for
the width of the block, and any thread that reads ``sys.stdout`` before the block
ends and writes to it after gets ``ValueError: I/O operation on closed file``.

That killed abstractinator-m at its first step: the snapshot publisher requested
the spec payload (which printed the model repr under a redirect) at the same
moment the compute profiler flushed stdout on the training thread. The profiler's
own error handler then used ``print``, failed identically, and escaped its
``except`` - turning optional telemetry into a fatal error.
"""

import io
import sys
import threading

import pytest


def test_capture_model_architecture_leaves_stdout_alone():
    """The spec payload must render the model without touching the global."""
    from praxis.web.spec_data import _capture_model_architecture

    class FakeModel:
        def __repr__(self):
            # Fails the test loudly if anyone reinstates the redirect: by the
            # time __repr__ runs, sys.stdout must still be the real one.
            assert sys.stdout is sentinel, "sys.stdout was swapped during repr"
            return "FakeModel(...)"

    class Gen:
        model = FakeModel()

    sentinel = sys.stdout
    out = _capture_model_architecture(Gen())
    assert "FakeModel" in out
    assert sys.stdout is sentinel


def test_profiler_quiet_logs_survives_a_closed_stdout():
    """The fd-level redirect is the point; a broken sys.stdout must not raise."""
    from praxis.metrics.compute import _quiet_native_logs

    original = sys.stdout
    closed = io.StringIO()
    closed.close()
    sys.stdout = closed
    try:
        with _quiet_native_logs():
            pass  # must not raise
    finally:
        sys.stdout = original


def test_profiler_callback_logger_never_raises():
    """Its whole job is to report failures, so it must not become one."""
    from praxis.callbacks.lightning.compute_profiler import _log_quietly

    original = sys.stdout
    closed = io.StringIO()
    closed.close()
    sys.stdout = closed
    try:
        _log_quietly("this must not raise")
    finally:
        sys.stdout = original


def test_no_background_thread_redirects_stdout():
    """A grep-level guard: redirect_stdout must not reappear off the main thread.

    paper.py is the known remaining offender - ``_build`` runs in a daemon
    thread and holds the global for an entire LaTeX build - so it is listed
    explicitly rather than silently tolerated.
    """
    import pathlib
    import re

    known = {"praxis/callbacks/lightning/paper.py"}
    root = pathlib.Path(__file__).resolve().parent.parent
    # CALLS only - the fix in spec_data.py names the hazard in a comment, and
    # matching prose would make this test un-passable by documenting itself.
    call = re.compile(r"^(?!\s*#).*\bredirect_std(out|err)\s*\(", re.M)
    offenders = set()
    for path in (root / "praxis").rglob("*.py"):
        if call.search(path.read_text()):
            offenders.add(str(path.relative_to(root)))
    assert offenders <= known, (
        f"new global-stdout redirect(s): {sorted(offenders - known)}. "
        "Render to a string instead; see praxis/web/spec_data.py."
    )
