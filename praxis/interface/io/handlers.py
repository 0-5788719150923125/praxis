"""Logging handlers for dashboard integration."""

import logging


class DashboardStreamHandler(logging.StreamHandler):
    """Stream handler that routes logs to the dashboard."""

    def __init__(self, dashboard):
        super().__init__()
        self.dashboard = dashboard

    def emit(self, record):
        msg = self.format(record)
        self.dashboard.add_log(msg)
