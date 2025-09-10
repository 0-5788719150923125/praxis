"""Dashboard output wrapper."""


class DashboardOutput:
    """Wrapper for original stdout to maintain compatibility."""
    
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout

    def write(self, data):
        self.original_stdout.write(data)

    def flush(self):
        self.original_stdout.flush()