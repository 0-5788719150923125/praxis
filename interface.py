import sys
import io
import time
import blessed
import asciichartpy
from collections import deque
from threading import Thread, Lock
import textwrap
import wcwidth
import logging
import sys
import io
import os


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
    def __init__(self, max_data_points=50, max_log_lines=100):
        self.term = blessed.Terminal()
        self.max_data_points = max_data_points
        self.train_losses = deque(maxlen=max_data_points)
        self.val_losses = deque(maxlen=max_data_points)
        self.status_text = "Initializing..."
        self.log_buffer = deque(maxlen=max_log_lines)
        self.step = 0
        self.running = False
        self.lock = Lock()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.dashboard_output = DashboardOutput(self.original_stdout)
        self.log_capture = LogCapture(self)
        self.previous_frame = None

    def start(self):
        self.running = True
        sys.stdout = self.log_capture
        sys.stderr = self.log_capture
        Thread(target=self._run_dashboard).start()

    def stop(self):
        self.running = False
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def update_losses(self, train_loss, val_loss):
        with self.lock:
            self.train_losses.append(train_loss) if train_loss is not None else None
            self.val_losses.append(val_loss) if val_loss is not None else None

    def update_step(self, step):
        with self.lock:
            self.step = step

    def update_status(self, status):
        with self.lock:
            self.status_text = status

    def add_log(self, message):
        with self.lock:
            # Split the message into lines, filter out empty lines, and strip whitespace
            new_lines = [line.strip() for line in message.splitlines() if line.strip()]
            self.log_buffer.extend(new_lines)

    def update_placeholder(self, text):
        with self.lock:
            self.placeholder_text = text

    def _draw_chart(self, data, width, height):
        if len(data) > 1:
            chart = asciichartpy.plot(
                list(data), {"height": height - 2, "width": width - 2}
            )
            return chart.split("\n")
        return []

    def _truncate_to_width(self, text, width):
        """Truncate text to fit within a given width, accounting for wide characters."""
        current_width = 0
        for i, char in enumerate(text):
            char_width = wcwidth.wcwidth(char)
            if current_width + char_width > width:
                return text[:i]
            current_width += char_width
        return text

    def _wrap_text(self, text, width):
        wrapped_lines = []
        for line in text.splitlines():
            wrapped_lines.extend(
                textwrap.wrap(
                    line, width=width, replace_whitespace=False, drop_whitespace=False
                )
            )
        return [
            self._truncate_to_width(line, width).ljust(width) for line in wrapped_lines
        ]

    def _draw_chart(self, data, width, height):
        if len(data) > 1:
            chart = asciichartpy.plot(
                list(data), {"height": height - 2, "width": width - 2}
            )
            return [line.ljust(width) for line in chart.split("\n")]
        return [" " * width for _ in range(height)]

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

        status_lines = self._wrap_text(self.status_text, right_width)
        log_lines = self._wrap_text(
            "\n".join(list(self.log_buffer)[-half_height + 3 :]), right_width
        )

        status_lines += [" " * right_width] * (half_height - 3 - len(status_lines))
        log_lines += [" " * right_width] * (half_height - 3 - len(log_lines))

        for i in range(height):
            left_content = " " * half_width
            right_content = " " * right_width

            if i == 0:
                left_content = " Training Loss".ljust(half_width)
                right_content = " Feed".ljust(right_width)
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
                left_content = " Validation Loss".ljust(half_width)
                right_content = " Logger".ljust(right_width)
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

            frame.append(f"║{left_content}║{right_content}║")

        frame.append("╚" + "═" * half_width + "╩" + "═" * right_width + "╝")

        with self.lock:
            train_loss = self.train_losses[-1] if self.train_losses else 0
            val_loss = self.val_losses[-1] if self.val_losses else 0
            frame.append(
                f" PRAXIS | Step: {self.step}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        return frame

    def _update_screen(self, new_frame):
        if self.previous_frame is None:
            print(
                self.term.home + self.term.clear + "\n".join(new_frame),
                end="",
                file=self.dashboard_output,
            )
        else:
            for i, (old_line, new_line) in enumerate(
                zip(self.previous_frame, new_frame)
            ):
                if old_line != new_line:
                    print(
                        self.term.move(i, 0) + new_line,
                        end="",
                        file=self.dashboard_output,
                    )

        self.previous_frame = new_frame
        self.dashboard_output.flush()

    def _run_dashboard(self):
        with self.term.fullscreen(), self.term.cbreak(), self.term.hidden_cursor():
            while self.running:
                new_frame = self._create_frame()
                self._update_screen(new_frame)
                time.sleep(0.1)


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
            dashboard.update_step(step)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop()
