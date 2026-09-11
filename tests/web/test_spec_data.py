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

import sys

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
