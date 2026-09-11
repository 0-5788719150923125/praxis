"""Spec payload (praxis/web/spec_data.py): rendering the model repr must not
swap the process-global ``sys.stdout``.

A ``redirect_stdout`` on the API thread redirects every other thread for the
width of the block; the training thread flushing stdout inside it hit a closed
file and killed abstractinator-m at its first step.
"""

import sys

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
