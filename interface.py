import atexit
import io
import logging
import os
import re
import signal
import sys
import textwrap
import time
import warnings
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timedelta
from threading import Lock, Thread

import asciichartpy
import blessed
import wcwidth


class DashboardStreamHandler(logging.StreamHandler):
    def __init__(self, dashboard):
        super().__init__()
        self.dashboard = dashboard

    def emit(self, record):
        msg = self.format(record)
        self.dashboard.add_log(msg)


class DashboardOutput:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout

    def write(self, data):
        self.original_stdout.write(data)

    def flush(self):
        self.original_stdout.flush()


class LogCapture:
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.log_buffer = io.StringIO()

    def write(self, data):
        self.log_buffer.write(data)
        self.dashboard.add_log(data)

    def flush(self):
        self.log_buffer.flush()


class TerminalDashboard:
    def __init__(self, seed, max_data_points=50, max_log_lines=100):
        self.seed = seed
        self.term = blessed.Terminal()
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        self.max_data_points = max_data_points
        self.train_losses = deque(maxlen=max_data_points)
        self.val_losses = deque(maxlen=max_data_points)
        self.status_text = "Initializing..."
        self.log_buffer = deque(maxlen=max_log_lines)
        self.batch = 0
        self.step = 0
        self.running = False
        self.lock = Lock()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.dashboard_output = DashboardOutput(self.original_stdout)
        self.log_capture = LogCapture(self)
        self.previous_frame = None
        self.start_time = datetime.now()
        self.url = "N/A"
        self.experts = ""
        self.total_params = ""

        # Set up logging
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.ERROR)
        handler = DashboardStreamHandler(self)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Capture warnings
        warnings.showwarning = self.show_warning

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Register the cleanup function
        atexit.register(self._cleanup)

    def show_warning(self, message, category, filename, lineno, file=None, line=None):
        warning_message = warnings.formatwarning(
            message, category, filename, lineno, line
        )
        self.add_log(warning_message)

    def _signal_handler(self, signum, frame):
        self.stop()
        sys.exit(0)

    def _cleanup(self):
        self.stop()
        self._reset_terminal()

    def _reset_terminal(self):
        print(
            self.term.normal
            + self.term.clear
            + self.term.home
            + self.term.visible_cursor,
            end="",
        )
        sys.stdout.flush()

    @contextmanager
    def managed_terminal(self):
        try:
            with self.term.fullscreen(), self.term.cbreak(), self.term.hidden_cursor():
                yield
        finally:
            self._reset_terminal()

    def start(self):
        self.running = True
        sys.stdout = self.log_capture
        sys.stderr = self.log_capture
        Thread(target=self._run_dashboard).start()

    def stop(self):
        self.running = False
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def update_seed(self, seed):
        with self.lock:
            self.seed = seed

    def update_params(self, total_params):
        with self.lock:
            reduced = int(total_params / 10**6)
            self.total_params = f" | {reduced}M"

    def update_losses(self, train_loss, val_loss):
        with self.lock:
            self.train_losses.append(train_loss) if train_loss is not None else None
            self.val_losses.append(val_loss) if val_loss is not None else None

    def update_step(self, step):
        with self.lock:
            self.step = step

    def update_batch(self, batch):
        with self.lock:
            self.batch = batch

    def update_url(self, url):
        with self.lock:
            self.url = url

    def update_utilization(self, utilization_percentages):
        with self.lock:
            self.experts = ", Experts: " + (
                "[ "
                + ", ".join(
                    f"'{i}': {percentage:.1f}%"
                    for i, percentage in enumerate(utilization_percentages)
                )
                + " ]"
            )

    def add_log(self, message):
        with self.lock:
            # Strip ANSI codes and split the message into lines
            stripped_message = self._strip_ansi(message)
            new_lines = [
                line.strip() for line in stripped_message.splitlines() if line.strip()
            ]
            self.log_buffer.extend(new_lines)

    def _strip_ansi(self, text):
        """Remove ANSI escape sequences from the text."""
        return self.ansi_escape.sub("", text)

    def hours_since(self):
        current_time = datetime.now()
        time_difference = current_time - self.start_time
        hours = time_difference.total_seconds() / 3600
        return hours

    def update_placeholder(self, text):
        with self.lock:
            self.placeholder_text = text

    def _visual_ljust(self, string, width):
        """Left-justify a string to a specified width, considering character display width."""
        visual_width = sum(wcwidth.wcwidth(char) for char in string)
        padding = max(0, width - visual_width)
        return string + " " * padding

    def _truncate_to_width(self, text, width):
        """Truncate text to fit within a given width, accounting for wide characters."""
        current_width = 0
        result = []
        for char in text:
            char_width = wcwidth.wcwidth(char)
            if current_width + char_width > width:
                break
            result.append(char)
            current_width += char_width
        return "".join(result)

    def update_status(self, status):
        with self.lock:
            self.status_text = status  # Keep this as it was originally

    def _create_frame(self):
        height = self.term.height - 4
        width = self.term.width - 3
        half_height = height // 2
        half_width = width // 2
        right_width = width - half_width - 1

        frame = []
        frame.append("╔" + "═" * half_width + "╦" + "═" * right_width + "╗")

        with self.lock:
            train_chart = self._draw_chart(
                self.train_losses, half_width, half_height - 1
            )
            val_chart = self._draw_chart(self.val_losses, half_width, half_height - 1)

        # Wrap the entire status text
        status_lines = self._wrap_text(self.status_text, right_width)

        # Calculate the maximum number of lines that can fit in the status section
        max_status_lines = half_height - 3

        # If status_lines exceed max_status_lines, keep only the most recent lines
        if len(status_lines) > max_status_lines:
            status_lines = status_lines[-max_status_lines:]

        log_text = "\n".join(list(self.log_buffer)[-half_height + 3 :])
        log_lines = self._wrap_text(log_text, right_width)

        # Pad status_lines and log_lines if they're shorter than the available space
        status_lines += [""] * (max_status_lines - len(status_lines))
        log_lines += [""] * (half_height - 3 - len(log_lines))

        for i in range(height):
            left_content = " " * half_width
            right_content = " " * right_width

            if i == 0:
                train_loss = self.train_losses[-1] if self.train_losses else 0
                left_content = self._visual_ljust(
                    f" Training Loss: {train_loss:.4f}", half_width
                )
                right_content = self._visual_ljust(" Feed", right_width)
            elif i == 1:
                left_content = "─" * half_width
                right_content = "─" * right_width
            elif i < half_height - 1:
                if i - 2 < len(train_chart):
                    left_content = train_chart[i - 2]
                if i - 2 < len(status_lines):
                    right_content = status_lines[i - 2]
            elif i == half_height - 1:
                left_content = "═" * half_width
                right_content = "═" * right_width
            elif i == half_height:
                val_loss = self.val_losses[-1] if self.val_losses else 0
                left_content = self._visual_ljust(
                    f" Frequency Bias: {val_loss:.4f}", half_width
                )
                right_content = self._visual_ljust(" Logger", right_width)
            elif i == half_height + 1:
                left_content = "─" * half_width
                right_content = "─" * right_width
            elif i > half_height + 1:
                chart_index = i - half_height - 2
                if chart_index < len(val_chart):
                    left_content = val_chart[chart_index]
                log_index = i - half_height - 2
                if log_index < len(log_lines):
                    right_content = log_lines[log_index]

            # Ensure left and right content are exactly the right width and strip ANSI codes
            left_content = self._strip_ansi(
                self._truncate_to_width(left_content, half_width)
            )
            right_content = self._strip_ansi(
                self._truncate_to_width(right_content, right_width)
            )
            left_content = self._visual_ljust(left_content, half_width)
            right_content = self._visual_ljust(right_content, right_width)

            frame.append(f"║{left_content}║{right_content}║")

        frame.append("╚" + "═" * half_width + "╩" + "═" * right_width + "╝")

        with self.lock:
            elapsed = self.hours_since()
            frame.append(
                self._truncate_to_width(
                    f"\n PRAXIS{str(self.total_params)}:{str(self.seed)} | Batch: {int(self.batch)}, Step: {int(self.step)} | Elapsed: {elapsed:.2f}h | URL: {self.url}",
                    width,
                )
            )

        return frame

    def _wrap_text(self, text, width):
        """Wrap text to fit within a given width, handling newlines and extra whitespace."""
        wrapped_lines = []
        for line in text.splitlines():
            # Strip leading/trailing whitespace from each line
            line = line.strip()
            if not line:
                wrapped_lines.append("")
                continue
            # Use textwrap to handle word wrapping
            wrapped = textwrap.wrap(
                line, width=width, break_long_words=True, replace_whitespace=False
            )
            wrapped_lines.extend(wrapped or [""])
        return wrapped_lines

    def _draw_chart(self, data, width, height):
        if len(data) > 1:
            # Ensure we only plot the most recent data points that fit in the width
            plot_data = list(data)[-width:]
            chart = asciichartpy.plot(
                plot_data,
                {
                    "height": height - 2,
                    "width": width - 2,
                    "format": "{:8.2f}",
                    "min": min(plot_data),
                    "max": max(plot_data),
                },
            )
            lines = chart.split("\n")
            # Ensure each line is exactly the right width
            return [line.ljust(width)[:width] for line in lines]
        return [" " * width for _ in range(height)]

    def _update_screen(self, new_frame):
        if self.previous_frame is None:
            print(
                self.term.home
                + self.term.clear
                + self.term.white
                + "\n".join(new_frame),
                end="",
                file=self.dashboard_output,
            )
        else:
            for i, (old_line, new_line) in enumerate(
                zip(self.previous_frame, new_frame)
            ):
                if old_line != new_line:
                    print(
                        self.term.move(i, 0) + self.term.white + new_line,
                        end="",
                        file=self.dashboard_output,
                    )

        self.previous_frame = new_frame
        self.dashboard_output.flush()

    def _run_dashboard(self):
        with self.managed_terminal():
            while self.running:
                try:
                    new_frame = self._create_frame()
                    self._update_screen(new_frame)
                    time.sleep(0.1)
                except Exception as e:
                    self.add_log(f"Dashboard error: {str(e)}")


# Example usage
if __name__ == "__main__":
    import random

    dashboard = TerminalDashboard()
    dashboard.start()

    step = 0
    try:
        for epoch in range(100):
            step += 1
            train_loss = 1 / (epoch + 1) + random.uniform(0, 0.1)
            val_loss = train_loss + random.uniform(0, 0.05)
            dashboard.update_losses(train_loss, val_loss)
            dashboard.update_status(f"Training... Epoch {epoch}")
            dashboard.update_batch(batch)
            dashboard.update_step(step)

            # Test logging at different levels
            dashboard.logger.debug("This is a debug message")
            dashboard.logger.info("This is an info message")
            dashboard.logger.warning("This is a warning message")
            dashboard.logger.error("This is an error message")

            # Test warnings
            warnings.warn("This is a test warning")

            # Test exceptions
            if epoch % 10 == 0:
                try:
                    raise ValueError("This is a test exception")
                except Exception as e:
                    dashboard.logger.exception("Caught an exception:")

            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop()
