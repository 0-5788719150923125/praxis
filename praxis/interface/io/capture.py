"""Log capture for dashboard display."""

import io


class LogCapture:
    """Captures stdout/stderr output for dashboard display."""
    
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.log_buffer = io.StringIO()

    def write(self, data):
        # Simply log everything to the LOG panel
        # The dashboard will only crash if the process itself crashes
        # (handled by signal handlers and exception hooks)
        self.log_buffer.write(data)
        self.dashboard.add_log(data.rstrip())

    def flush(self):
        self.log_buffer.flush()