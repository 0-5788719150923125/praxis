"""Log capture for dashboard display."""

import sys


class LogCapture:
    """Stands in for stdout/stderr, routing writes into the LOG panel.

    Third-party code inspects the stream it was handed (``isatty`` to decide
    whether to draw a progress bar, ``encoding`` to encode bytes, ``fileno`` to
    reach the descriptor). A bare object with only ``write``/``flush`` either
    raises on those or - worse - lets a library conclude it owns a TTY and start
    emitting cursor movement into the log panel. The stream protocol below is
    the redirection contract: it reports "not a terminal", and it delegates the
    rest to the real stream underneath.
    """

    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.errors = "replace"

    # ── stream protocol ──────────────────────────────────────────────────

    @property
    def _original(self):
        return getattr(self.dashboard, "original_stdout", sys.__stdout__)

    @property
    def encoding(self):
        # TextIOBase defines `encoding` as None, so a plain getattr default is
        # not enough - callers that encode with it would get a TypeError.
        return getattr(self._original, "encoding", None) or "utf-8"

    @property
    def closed(self):
        return False

    def isatty(self):
        # Deliberately False: nothing writing through the dashboard should be
        # drawing progress bars or repositioning a cursor we own.
        return False

    def fileno(self):
        # Some callers only want a descriptor to hand to a subprocess; give
        # them the real one rather than raising.
        return self._original.fileno()

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False

    # ── writes ───────────────────────────────────────────────────────────

    def write(self, data):
        # The dashboard decides where this goes; once it has released the
        # screen, add_log forwards to the real stderr instead of a buffer
        # nobody will ever render.
        self.dashboard.add_log(data.rstrip())
        return len(data)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        pass
