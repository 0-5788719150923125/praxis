"""Activity monitoring for dashboard."""

import time


class ActivityMonitor:
    """Monitors activity and detects inactivity periods."""

    def __init__(self, warning_timeout=60):
        self.last_activity = time.time()
        self.warning_timeout = warning_timeout
        self.last_warning_time = 0

    def mark_activity(self):
        """Mark that we've received activity from the training process."""
        self.last_activity = time.time()

    def check_inactivity(self):
        """Check for inactivity and return warning if needed."""
        current_time = time.time()
        inactive_time = current_time - self.last_activity

        if (
            inactive_time > self.warning_timeout
            and current_time - self.last_warning_time > 300
        ):  # Warn max once per 5 minutes
            self.last_warning_time = current_time
            return inactive_time
        return None
