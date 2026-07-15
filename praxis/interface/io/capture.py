"""Log capture for dashboard display."""


class LogCapture:
    """Captures stdout/stderr output for dashboard display."""

    def __init__(self, dashboard):
        self.dashboard = dashboard

    def write(self, data):
        # Simply log everything to the LOG panel
        # The dashboard will only crash if the process itself crashes
        # (handled by signal handlers and exception hooks)
        self.dashboard.add_log(data.rstrip())

    def flush(self):
        pass
