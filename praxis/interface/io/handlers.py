"""Logging handlers for dashboard integration."""

import logging


class DashboardStreamHandler(logging.StreamHandler):
    """Stream handler that routes logs to the dashboard.

    The dashboard strips every other handler off every logger when it starts,
    so this is the only path records have. It must therefore keep working after
    the dashboard releases the screen - otherwise everything logged during
    shutdown (the part of a run you most want to read) is written into a buffer
    that will never be rendered. ``add_log`` handles that fallback.
    """

    def __init__(self, dashboard):
        super().__init__()
        self.dashboard = dashboard

    def emit(self, record):
        try:
            msg = self.format(record)
            self.dashboard.add_log(msg)
        except Exception:
            self.handleError(record)
