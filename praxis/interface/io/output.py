"""Dashboard output wrapper."""


class DashboardOutput:
    """The painter's private handle on the real stdout.

    The render thread writes frames here, bypassing the stdout redirection the
    dashboard installs for everyone else. That bypass is what made shutdown
    unsafe: once the terminal leaves fullscreen, a frame written through this
    handle lands in the user's scrollback as absolute-positioned box drawing.
    ``disable()`` closes the tap, so a paint that is already in flight when the
    screen is released cannot reach the terminal; ``start()`` reopens it when the
    dashboard claims a screen again.
    """

    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.enabled = True

    def disable(self):
        """Stop forwarding writes to the terminal."""
        self.enabled = False

    def enable(self):
        """Reopen the tap - only when a fresh screen has been claimed."""
        self.enabled = True

    def write(self, data):
        if not self.enabled:
            return len(data)
        return self.original_stdout.write(data)

    def flush(self):
        if not self.enabled:
            return
        self.original_stdout.flush()
