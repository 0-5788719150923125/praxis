import asyncio
import atexit
import io
import logging
import os
import random
import re
import shutil
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
import numpy as np
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
        # Simply log everything to the LOG panel
        # The dashboard will only crash if the process itself crashes
        # (handled by signal handlers and exception hooks)
        self.log_buffer.write(data)
        self.dashboard.add_log(data.rstrip())

    def flush(self):
        self.log_buffer.flush()


class TerminalDashboard:
    def __init__(
        self, seed, arg_hash="000000", max_data_points=1000, max_log_lines=200
    ):
        self.seed = seed
        self.term = blessed.Terminal()
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        self.max_data_points = max_data_points
        self.train_losses = deque(maxlen=max_data_points)
        self.sign = 1
        self.val_loss = None
        self.status_text = "_initializing"

        # Save terminal state for proper restoration
        self.saved_terminal_state = None
        self.terminal_restored = False

        # Activity monitoring (for warnings, not force-crash)
        self.last_activity = time.time()
        self.warning_timeout = 60  # Warn after 60 seconds of no activity
        self.last_warning_time = 0  # Track when we last warned
        self.log_buffer = deque(maxlen=max_log_lines)
        self.batch = 0
        self.step = 0
        self.running = False
        self.lock = Lock()
        self.previous_size = self._get_terminal_size()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.dashboard_output = DashboardOutput(self.original_stdout)
        self.log_capture = LogCapture(self)
        self.previous_frame = None
        self.start_time = datetime.now()
        self.rate = 0
        self.url = "N/A"
        self.total_params = "0M"
        self.mode = "train"
        self.local_experts = 0
        self.remote_experts = 0
        self.fitness = None
        self.memory_churn = None
        self.accuracy = None
        self.num_tokens = 0
        self.game_of_life = None
        self.info_dict = {}

        self.arg_hash = arg_hash

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
        self.error_exit = False
        self.error_message = None
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
        # Only cleanup if we haven't already done so
        if self.running:
            self.stop()
        # Always try to restore terminal on cleanup
        if not self.terminal_restored:
            self._restore_terminal()
            self.terminal_restored = True

    def _restore_terminal(self):
        """Fully restore terminal to its original state."""
        try:
            # First exit fullscreen and restore cursor
            if hasattr(self, "original_stderr"):
                # Exit fullscreen mode (this restores the original terminal content)
                print(self.term.exit_fullscreen, end="", file=self.original_stderr)
                # Reset all terminal attributes
                print(self.term.normal, end="", file=self.original_stderr)
                # Make cursor visible
                print(self.term.visible_cursor, end="", file=self.original_stderr)
                # Don't clear or home - this preserves terminal history!
                # Just ensure we're on a new line for clean output
                print("", file=self.original_stderr)
                self.original_stderr.flush()

            # Restore saved terminal settings if available
            if self.saved_terminal_state is not None:
                try:
                    import termios

                    if hasattr(sys.stdin, "fileno"):
                        termios.tcsetattr(
                            sys.stdin.fileno(),
                            termios.TCSANOW,
                            self.saved_terminal_state,
                        )
                except:
                    pass

        except Exception:
            # If anything fails, at least try to make the terminal usable
            try:
                sys.stderr.write("\033[0m\033[?25h\n")  # Reset and show cursor
                sys.stderr.flush()
            except:
                pass

    def _get_terminal_size(self):
        return shutil.get_terminal_size()

    @contextmanager
    def managed_terminal(self):
        try:
            with self.term.fullscreen(), self.term.cbreak(), self.term.hidden_cursor():
                yield
        finally:
            # Ensure terminal is restored when exiting the context
            if not self.terminal_restored and not self.error_exit:
                self._restore_terminal()
                self.terminal_restored = True

    def start(self):
        self.running = True

        # Save terminal state before modifying it
        try:
            # Save current terminal settings
            import termios
            import tty

            if hasattr(sys.stdin, "fileno"):
                try:
                    self.saved_terminal_state = termios.tcgetattr(sys.stdin.fileno())
                except:
                    self.saved_terminal_state = None
        except ImportError:
            self.saved_terminal_state = None

        # Set up signal handlers to catch crashes
        self._setup_signal_handlers()

        # Capture stdout and stderr
        sys.stdout = self.log_capture
        sys.stderr = self.log_capture

        # Force unbuffered output for better error capture
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(line_buffering=True)

        # Set environment variables to help with CUDA error reporting
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Make CUDA errors appear immediately
        # os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'  # Show C++ stack traces

        Thread(target=self._run_dashboard).start()

    def _setup_signal_handlers(self):
        """Set up signal handlers to catch crashes and display errors immediately."""

        def crash_handler(signum, frame):
            # Generate a stack trace from the crash point
            import traceback

            stack_trace = "".join(traceback.format_stack(frame))
            error_text = (
                f"Process crashed with signal {signum}\n\nStack trace:\n{stack_trace}"
            )
            self.crash_with_error(error_text)

        # Handle various crash signals
        try:
            signal.signal(signal.SIGSEGV, crash_handler)  # Segmentation fault
            signal.signal(signal.SIGFPE, crash_handler)  # Floating point exception
            signal.signal(signal.SIGILL, crash_handler)  # Illegal instruction
            signal.signal(signal.SIGABRT, crash_handler)  # Abort signal
        except (OSError, ValueError):
            # Some signals may not be available on all platforms
            pass

        # Handle Ctrl+C gracefully
        def keyboard_interrupt_handler(signum, frame):
            # Ensure terminal is restored before exiting
            if not self.terminal_restored:
                self._restore_terminal()
                self.terminal_restored = True
            self.stop()
            print("\n🛑 Training interrupted by user", file=sys.stderr)
            sys.exit(0)

        signal.signal(signal.SIGINT, keyboard_interrupt_handler)

        # Set up global exception handler for uncaught exceptions
        def exception_handler(exc_type, exc_value, exc_traceback):
            if exc_type == KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                if not self.terminal_restored:
                    self._restore_terminal()
                    self.terminal_restored = True
                self.stop()
                print("\n🛑 Training interrupted by user", file=sys.stderr)
                sys.exit(0)
            else:
                # Format the exception - this is a real crash
                import traceback

                error_lines = traceback.format_exception(
                    exc_type, exc_value, exc_traceback
                )
                error_text = "".join(error_lines)

                # Also capture recent log output for context
                recent_logs = ""
                if hasattr(self, "log_buffer") and self.log_buffer:
                    recent_logs = "\n\nRecent log output:\n" + "-" * 40 + "\n"
                    recent_logs += "\n".join(
                        list(self.log_buffer)[-20:]
                    )  # Last 20 lines

                self.crash_with_error(error_text + recent_logs)

        sys.excepthook = exception_handler

    def stop(self, error=False):
        self.running = False
        self.error_exit = error

        # Restore stdout/stderr first
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Perform full terminal restoration if not already done
        if not self.terminal_restored:
            self._restore_terminal()
            self.terminal_restored = True

    def crash_with_error(self, error_text):
        """Immediately crash the dashboard and display the error."""
        # Restore original stdout/stderr first
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Force stop the dashboard
        self.running = False
        self.error_exit = True

        # CRITICAL: Restore terminal properly
        if not self.terminal_restored:
            self._restore_terminal()
            self.terminal_restored = True

        # Small delay to ensure terminal mode switch completes
        import time

        time.sleep(0.1)

        # Now show the error in normal terminal mode where scrolling works
        # Add some newlines to separate from any dashboard artifacts
        print("\n\n", file=sys.stderr)
        print(
            "🚨 TRAINING CRASHED - Dashboard terminated to show error:", file=sys.stderr
        )
        print("=" * 80, file=sys.stderr)

        # Make sure we have some error text to display
        if not error_text or error_text.strip() == "":
            error_text = "Unknown error occurred (no error message captured)"

        print(error_text, file=sys.stderr)
        print("=" * 80, file=sys.stderr)

        # Store for potential later use
        self.error_message = error_text

        # Flush to ensure output is visible
        sys.stderr.flush()

        # Exit immediately - don't wait for dashboard thread
        import os

        os._exit(1)

    def _mark_activity(self):
        """Mark that we've received activity from the training process."""
        self.last_activity = time.time()

    def update_seed(self, seed):
        with self.lock:
            self.seed = seed

    def set_mode(self, mode="train"):
        with self.lock:
            self.mode = mode
            self.force_redraw()

    def set_start_time(self, time):
        with self.lock:
            self.start_time = time

    def force_redraw(self):
        self.previous_frame = None

    def update_status(self, status):
        with self.lock:
            self.status_text = status

    def update_status_internal(self, status):
        """Internal status update (no lock needed - caller handles locking)."""
        self.status_text = status

    def update_params(self, total_params):
        with self.lock:
            reduced = int(total_params / 10**6)
            self.total_params = f"{reduced}M"
            self.previous_frame = None  # force a redraw

    def update_loss(self, train_loss):
        with self.lock:
            self._mark_activity()
            self.train_losses.append(train_loss) if train_loss is not None else None

    def update_val(self, val_loss):
        with self.lock:
            self.val_loss = val_loss

    def update_accuracy(self, acc0, acc1):
        with self.lock:
            self.accuracy = [acc0, acc1]

    def update_fitness(self, fitness):
        with self.lock:
            self.fitness = fitness

    def update_memory(self, churn):
        with self.lock:
            self.memory_churn = churn

    def update_step(self, step):
        with self.lock:
            self._mark_activity()
            self.step = step

    def update_batch(self, batch):
        with self.lock:
            self._mark_activity()
            self.batch = batch

    def update_rate(self, seconds):
        with self.lock:
            self.rate = seconds

    def update_tokens(self, num_tokens):
        with self.lock:
            self.num_tokens = num_tokens

    def update_expert_count(self, num_local, num_remote):
        with self.lock:
            self.local_experts = num_local
            self.remote_experts = num_remote

    def update_url(self, url):
        with self.lock:
            self.url = url

    def update_info(self, info_dict):
        """Update the key/value pairs to display in the info panel."""
        with self.lock:
            self.info_dict = {**self.info_dict, **info_dict}

    def add_log(self, message):
        with self.lock:
            # Strip ANSI codes and split the message into lines
            stripped_message = self._strip_ansi(message)
            # Don't strip lines - preserve them as-is, but filter empty ones
            new_lines = [line for line in stripped_message.splitlines() if line]
            # Handle single line messages without newlines
            if not new_lines and stripped_message.strip():
                new_lines = [stripped_message.strip()]
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

    def _sanitize_text(self, text):
        """
        Sanitize text by replacing problematic characters with safe alternatives.
        Returns sanitized text with consistent character widths.
        """
        if not text:
            return ""

        result = []
        for char in text:
            width = wcwidth.wcwidth(char)
            if width < 0 or width > 1:  # Problematic character detected
                result.append("�")  # Using an emoji as a safe replacement
            else:
                result.append(char)

        return "".join(result)

    # Update the _truncate_to_width method to use sanitization
    def _truncate_to_width(self, text, width):
        """Truncate text to fit within a given width, accounting for wide characters."""
        if not text:
            return ""

        # Sanitize the input text first
        sanitized_text = self._sanitize_text(text)

        current_width = 0
        result = []
        for char in sanitized_text:
            char_width = wcwidth.wcwidth(char)
            if char_width < 0:
                char_width = 1  # Treat any remaining problematic characters as width 1
            if current_width + char_width > width:
                break
            result.append(char)
            current_width += char_width

        return "".join(result)

    # Update the _visual_ljust method to use sanitization
    def _visual_ljust(self, string, width):
        """Left-justify a string to a specified width, considering character display width."""
        if not string:
            return " " * width

        # Sanitize the input string first
        sanitized_string = self._sanitize_text(string)

        visual_width = sum(max(wcwidth.wcwidth(char), 0) for char in sanitized_string)
        padding = max(0, width - visual_width)
        return sanitized_string + " " * padding

    def _visual_len(self, s):
        """Calculate the visual display width of a string."""
        return sum(max(wcwidth.wcwidth(char), 0) for char in s)

    def _correct_borders(self, frame):
        frame_visual_width = self._visual_len(frame[0])
        for i in range(1, len(frame) - 1):
            line = frame[i]
            line_visual_len = self._visual_len(line)
            if line_visual_len < frame_visual_width:
                padding_needed = frame_visual_width - line_visual_len
                line += " " * padding_needed
            elif line_visual_len > frame_visual_width:
                line = self._truncate_to_width(line, frame_visual_width)
            if not line.startswith("║"):
                line = "║" + line[1:]
            if not line.endswith("║"):
                line = line[:-1] + "║"
            frame[i] = line
        return frame

    def _check_border_alignment(self, frame):
        # Assuming the ERROR section is on the first content line after the top border
        error_line_index = 1  # Adjust if necessary
        line = frame[error_line_index]
        expected_length = self._visual_len(frame[0])  # Length of the top border
        line_visual_len = self._visual_len(line)
        if line_visual_len != expected_length:
            return False
        if not line.startswith("║") or not line.endswith("║"):
            return False
        return True

    def _update_screen(self, new_frame):
        # Correct borders for all lines
        new_frame = self._correct_borders(new_frame)
        # No need to pad lines here; they should already be the correct length
        frame_width = len(new_frame[0])

        if self.previous_frame is None or len(self.previous_frame) != len(new_frame):
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

    def _create_frame(self):
        # Get current terminal size
        current_size = self._get_terminal_size()

        # Check if terminal size has changed
        if current_size != self.previous_size:
            self.previous_frame = None  # Force full redraw
            self.previous_size = current_size

        width, height = current_size
        height -= 4  # Adjust for status bar and borders
        width -= 3  # Adjust for left and right borders

        if width <= 0 or height <= 0:
            # Prevent negative or zero dimensions
            return [" " * current_size.columns for _ in range(current_size.lines)]

        half_width = width // 2
        right_width = width - half_width  # Correct calculation

        frame = []
        frame.append("╔" + "═" * half_width + "╦" + "═" * right_width + "╗")

        # Split the lower-left panel into two parts
        lower_left_quarter_width = half_width // 2
        lower_right_quarter_width = half_width - lower_left_quarter_width

        with self.lock:
            train_chart = self._draw_chart(
                self.train_losses, right_width, (height // 2) - 1
            )
            sim_chart = self._draw_simulation(
                lower_left_quarter_width, (height // 2) - 1
            )
            info_chart = self._draw_info(lower_right_quarter_width, (height // 2) - 1)

        # Wrap the entire status text
        status_lines = self._wrap_text(self.status_text, half_width)

        # Calculate the maximum number of lines that can fit in the status section
        max_status_lines = (height // 2) - 3

        # If status_lines exceed max_status_lines, keep only the most recent lines
        if len(status_lines) > max_status_lines:
            status_lines = status_lines[-max_status_lines:]

        log_text = "\n".join(list(self.log_buffer)[-((height // 2) - 1) :])
        log_lines = self._wrap_text(log_text, right_width)

        # Pad status_lines and log_lines if they're shorter than the available space
        status_lines += [""] * (max_status_lines - len(status_lines))
        log_lines += [""] * ((height // 2) - 1 - len(log_lines))

        for i in range(height):
            left_content = " " * half_width
            right_content = " " * right_width

            if i == 0:
                train_loss = self.train_losses[-1] if self.train_losses else 0
                text = f" ERROR: {train_loss:.4f}"
                if self.val_loss is not None:
                    text += f" || VALIDATION: {self.val_loss:.4f}"
                if self.fitness is not None:
                    text += f" || FITNESS: {self.fitness:.4f}%"
                if self.memory_churn is not None:
                    text += f" || SURPRISE: {self.memory_churn:.2f}%"
                if self.accuracy is not None:
                    text += f" || ACCURACY: {self.accuracy[0]:.3f} || CONFIDENCE: {self.accuracy[1]:.3f}"
                # Truncate before padding
                right_content = self._truncate_to_width(text, right_width)
                right_content = right_content.ljust(right_width)
                left_content = self._truncate_to_width(f" {self.arg_hash}", half_width)
                left_content = left_content.ljust(half_width)
            elif i == 1:
                left_content = "─" * half_width
                right_content = "─" * right_width
            elif i < (height // 2) - 1:
                if i - 2 < len(status_lines):
                    left_content = status_lines[i - 2]
                if i - 2 < len(train_chart):
                    right_content = train_chart[i - 2]
            elif i == (height // 2) - 1:
                left_content = "═" * half_width
                right_content = "═" * right_width
            elif i == (height // 2):
                if random.random() < 0.1:
                    self.sign = -1 * self.sign
                # Split the left section into two parts
                attention_label = f" ATTENTION: {self.sign:+.1f}"
                info_label = " INFO"
                attention_content = attention_label.ljust(lower_left_quarter_width)[
                    :lower_left_quarter_width
                ]
                info_content = info_label.ljust(lower_right_quarter_width)[
                    :lower_right_quarter_width
                ]
                left_content = attention_content + "║" + info_content
                right_content = " LOG".ljust(right_width)[:right_width]
            elif i == (height // 2) + 1:
                # Split the separator line for the lower left panel
                left_content = (
                    "─" * lower_left_quarter_width
                    + "╫"
                    + "─" * lower_right_quarter_width
                )
                right_content = "─" * right_width
            elif i > (height // 2) + 1:
                chart_index = i - (height // 2) - 2
                # Left side split into two parts
                sim_content = " " * lower_left_quarter_width
                info_content = " " * lower_right_quarter_width
                if chart_index < len(sim_chart):
                    sim_content = sim_chart[chart_index]
                if chart_index < len(info_chart):
                    info_content = info_chart[chart_index]

                # Ensure both parts are exactly the right width (same as header approach)
                sim_content = sim_content.ljust(lower_left_quarter_width)[
                    :lower_left_quarter_width
                ]
                info_content = info_content.ljust(lower_right_quarter_width)[
                    :lower_right_quarter_width
                ]
                left_content = sim_content + "║" + info_content

                log_index = i - (height // 2) - 2
                if log_index < len(log_lines):
                    right_content = log_lines[log_index]

            # Truncate and pad left content
            left_content = self._truncate_to_width(left_content, half_width)
            left_content = self._visual_ljust(left_content, half_width)

            # Truncate and pad right content
            right_content = self._truncate_to_width(right_content, right_width)
            right_content = self._visual_ljust(right_content, right_width)

            # Combine the content with borders
            frame.append(f"║{left_content}║{right_content}║")

        frame.append("╠" + "═" * half_width + "╩" + "═" * right_width + "╣")

        with self.lock:
            elapsed = self.hours_since()
            footer_text = (
                f" PRAXIS:{str(self.seed)} | {self.total_params} | MODE: {self.mode} | "
                f"AGE: {elapsed:.2f}h | TOKENS: {self.num_tokens:.2f}B | BATCH: {int(self.batch)}, STEP: {int(self.step)}, "
                f"RATE: {self.rate:.2f}s | {self.local_experts} local experts, "
                f"{self.remote_experts} remote | {self.url}"
            )
            # Truncate and pad the footer text to fit the width
            footer_text = self._truncate_to_width(footer_text, width + 1)
            footer_text = footer_text.ljust(width + 1)
            frame.append("║" + footer_text + "║")

        # Add bottom border
        frame.append("╚" + "═" * (width + 1) + "╝")

        return frame

    def _wrap_text(self, text, width):
        """Wrap text to fit within a given width, preserving newlines."""
        wrapped_lines = []
        for line in text.splitlines():
            if line == "":  # Handle explicit empty lines (newlines)
                wrapped_lines.append("")  # Just append an empty line
                continue
            # Wrap the text normally
            wrapped = textwrap.wrap(
                line, width=width, break_long_words=True, replace_whitespace=False
            )
            wrapped_lines.extend(wrapped)
        return wrapped_lines

    def _draw_chart(self, data, width, height):
        # For other charts, use the original implementation
        if len(data) > 1:
            # Use logarithmic time scale to show more recent data in detail
            # while still including historical context
            num_points = min(1000, len(data))
            all_data = list(data)[-num_points:]

            if len(all_data) <= width:
                # If we have fewer points than display width, just use them all
                plot_data = all_data
            else:
                # Create logarithmic sampling: more samples from recent data
                plot_data = []

                # We'll use a power function to distribute sample indices
                # Higher power = more bias towards recent data
                power = 2.0  # Adjust this to control the bias (1.0 = linear, higher = more recent bias)

                for i in range(width):
                    # Map display position to data index using power function
                    # i/width goes from 0 to 1, we transform it to sample more from the end
                    normalized_pos = i / (width - 1) if width > 1 else 0
                    # Apply power function to bias towards recent data
                    biased_pos = pow(normalized_pos, power)
                    # Map to data index
                    data_idx = int(biased_pos * (len(all_data) - 1))

                    # For smoother visualization, average a small window around this point
                    window_size = max(1, len(all_data) // width)
                    start_idx = max(0, data_idx - window_size // 2)
                    end_idx = min(len(all_data), start_idx + window_size)

                    window_data = all_data[start_idx:end_idx]
                    if window_data:
                        plot_data.append(sum(window_data) / len(window_data))

            # Add slight smoothing to make trends more visible
            if len(plot_data) > 3:
                smoothed_data = []
                for i in range(len(plot_data)):
                    # Simple moving average with small window
                    start = max(0, i - 1)
                    end = min(len(plot_data), i + 2)
                    smoothed_data.append(sum(plot_data[start:end]) / (end - start))
                plot_data = smoothed_data

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
            return [line.ljust(width)[:width] for line in lines]
        return [" " * width for _ in range(height)]

    def _draw_simulation(self, width, height):
        # Account for each cell being 2 characters wide, use complete height
        if (
            self.game_of_life is None
            or self.game_of_life.width != (width - 2) // 2
            or self.game_of_life.height != height
        ):
            self.game_of_life = ForestFireAutomata((width - 2) // 2, height)

        # Update the game state
        self.game_of_life.get_next_generation()

        # Convert to ASCII and pad to full width
        lines = self.game_of_life.to_ascii()
        # Minimal single-space padding for alignment
        return [" " + line + " " for line in lines]

    def _draw_info(self, width, height):
        """Draw the info panel with key/value pairs."""
        lines = []

        # Process items to handle lists that need multiple lines
        display_items = []
        for key, value in self.info_dict.items():
            if isinstance(value, list):
                # Convert list to string representation
                val_str = str(value)
                max_key_len = width // 3
                max_val_len = width - max_key_len - 3  # -3 for ": " and padding

                # If the list representation is too long, wrap it intelligently
                if len(val_str) > max_val_len:
                    # Smart wrapping that respects list structure
                    wrapped_parts = self._wrap_list_string(val_str, max_val_len)
                    # First line with key
                    display_items.append((key, wrapped_parts[0]))
                    # Continuation lines with empty key
                    for part in wrapped_parts[1:]:
                        display_items.append(("", part))
                else:
                    display_items.append((key, val_str))
            else:
                display_items.append((key, value))

        # Format each display item
        for i in range(height):
            if i < len(display_items):
                key, value = display_items[i]
                # Truncate key to fit
                max_key_len = width // 3
                max_val_len = width - max_key_len - 3  # -3 for ": " and padding

                key_str = str(key)[:max_key_len]
                val_str = str(value)[:max_val_len]

                if key:  # Normal line with key
                    line = f" {key_str}: {val_str}"
                else:  # Continuation line
                    line = f" {' ' * max_key_len}  {val_str}"

                lines.append(line.ljust(width)[:width])
            else:
                lines.append(" " * width)

        return lines

    def _wrap_list_string(self, list_str, max_width):
        """Wrap a list string representation intelligently, breaking on commas and spaces."""
        if len(list_str) <= max_width:
            return [list_str]

        wrapped = []
        current_line = ""

        # Try to break on commas followed by spaces
        i = 0
        while i < len(list_str):
            char = list_str[i]
            current_line += char

            # Check if we've reached the line limit
            if len(current_line) >= max_width:
                # Look for the last comma or space to break on
                break_point = -1

                # First try to find a comma followed by space
                for j in range(len(current_line) - 1, -1, -1):
                    if (
                        j > 0
                        and current_line[j - 1] == ","
                        and j < len(current_line)
                        and current_line[j] == " "
                    ):
                        break_point = j
                        break

                # If no comma+space found, try just comma
                if break_point == -1:
                    for j in range(len(current_line) - 1, -1, -1):
                        if current_line[j] == ",":
                            break_point = j + 1  # Keep comma on current line
                            break

                # If no comma found, try space (but not within quotes)
                if break_point == -1:
                    in_quotes = False
                    for j in range(len(current_line) - 1, -1, -1):
                        if current_line[j] in ['"', "'"]:
                            in_quotes = not in_quotes
                        elif current_line[j] == " " and not in_quotes:
                            break_point = j + 1  # Keep space on current line
                            break

                # If we found a break point, use it
                if break_point > 0:
                    wrapped.append(current_line[:break_point])
                    current_line = current_line[break_point:]
                else:
                    # No good break point found, just break at max width
                    wrapped.append(current_line)
                    current_line = ""

            i += 1

        # Add any remaining content
        if current_line:
            wrapped.append(current_line)

        return wrapped

    def _run_dashboard(self):
        try:
            with self.managed_terminal():
                while self.running:
                    try:
                        # Check for long periods of inactivity (warn but don't crash)
                        current_time = time.time()
                        inactive_time = current_time - self.last_activity

                        if (
                            inactive_time > self.warning_timeout
                            and current_time - self.last_warning_time > 300
                        ):  # Warn max once per 5 minutes
                            # Instead of adding to log (which breaks display), just update status
                            # This avoids the dashboard rendering issues entirely
                            # with self.lock:
                            #     self.update_status_internal(f"No activity for {int(inactive_time)}s (inference/eval?)")
                            self.last_warning_time = current_time

                        new_frame = self._create_frame()
                        if not self._check_border_alignment(new_frame):
                            self.previous_frame = None
                        self._update_screen(new_frame)
                        time.sleep(0.1)
                    except Exception as e:
                        # Don't just log dashboard errors - they might indicate a deeper problem
                        error_msg = f"Dashboard rendering error: {str(e)}"
                        self.add_log(error_msg)
                        # If we get repeated dashboard errors, something is seriously wrong
                        self.crash_with_error(f"Dashboard crashed with: {error_msg}")
                        return
        finally:
            # After exiting fullscreen mode, print any stored error
            if self.error_message:
                print(self.error_message, file=self.original_stderr)


class ForestFireAutomata:
    def __init__(self, width, height):
        """Initialize the forest fire simulation."""
        self.width = width
        self.height = height
        self.p_growth = 0.01
        self.p_lightning = 0.001

        # States: 0 = empty, 1 = tree, 2 = burning
        self.grid = np.zeros((height, width))
        self.grid = np.random.choice([0, 1], size=(height, width), p=[0.8, 0.2])

    def get_next_generation(self):
        new_grid = np.copy(self.grid)

        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 0:  # Empty
                    if np.random.random() < self.p_growth:
                        new_grid[i, j] = 1

                elif self.grid[i, j] == 1:  # Tree
                    neighbors = self.grid[
                        max(0, i - 1) : min(i + 2, self.height),
                        max(0, j - 1) : min(j + 2, self.width),
                    ]
                    if 2 in neighbors:
                        new_grid[i, j] = 2
                    elif np.random.random() < self.p_lightning:
                        new_grid[i, j] = 2

                elif self.grid[i, j] == 2:  # Burning
                    new_grid[i, j] = 0

        self.grid = new_grid
        return self.grid

    def to_ascii(self):
        """Convert the grid to ASCII art."""
        return [
            "".join(
                ["██" if cell == 1 else "░░" if cell == 2 else "  " for cell in row]
            )
            for row in self.grid
        ]


# Test text with various newline patterns
TEST_TEXT = """The implementation of artificial neural networks has revolutionized machine learning in recent years. Deep learning models have achieved unprecedented success in various tasks, from image recognition to natural language processing. The key to their success lies in their ability to learn hierarchical representations of data through multiple layers of processing.

This architectural approach allows for automatic feature extraction, eliminating the need for manual feature engineering that was previously required in traditional machine learning approaches.


Model training presents its own unique set of challenges. The optimization of neural network parameters requires careful consideration of learning rates, batch sizes, and initialization strategies. Additionally, the choice of activation functions can significantly impact model performance.

Sometimes small changes have big effects.



Gradient descent optimization remains a fundamental technique in deep learning. The process involves calculating partial derivatives with respect to each parameter in the network, enabling the model to adjust its weights in a direction that minimizes the loss function.

The backpropagation algorithm, essential for training deep neural networks, efficiently computes these gradients through the chain rule of calculus.


Regularization techniques play a crucial role in preventing overfitting:
1. Dropout randomly deactivates neurons during training
2. L1 and L2 regularization add penalty terms to the loss function
3. Batch normalization stabilizes the learning process

These methods help ensure the model generalizes well to unseen data.



The architecture of modern neural networks has grown increasingly complex. Transformer models, for instance, have revolutionized natural language processing through their self-attention mechanisms.

This innovation has led to breakthrough models like BERT, GPT, and their successors.


The computational requirements for training large models are substantial:
- High-performance GPUs or TPUs are often necessary
- Distributed training across multiple devices is common
- Memory optimization techniques are crucial

These requirements have driven advances in hardware acceleration and distributed computing.



Recent developments in few-shot learning and meta-learning have opened new possibilities. These approaches allow models to learn from limited examples, more closely mimicking human learning capabilities.

The field continues to evolve rapidly, with new architectures and training methods emerging regularly.


Ethical considerations in AI development have become increasingly important:
- Model bias and fairness
- Environmental impact of large-scale training
- Privacy concerns with data usage

These issues require careful consideration from researchers and practitioners.



The future of deep learning looks promising, with potential applications in:
1. Medical diagnosis and treatment
2. Climate change modeling
3. Autonomous systems
4. Scientific discovery

Each application brings its own unique challenges and opportunities.

The intersection of deep learning with other fields continues to yield interesting results. Quantum computing, for instance, may offer new approaches to optimization problems in neural network training.


This ongoing evolution of the field requires continuous learning and adaptation from practitioners. The rapid pace of development means that today's state-of-the-art might be outdated within months.

Best practices and methodologies must therefore remain flexible and adaptable.



The role of benchmarking and evaluation metrics cannot be overstated. Proper evaluation of model performance requires careful consideration of various metrics:
- Accuracy and precision
- Recall and F1 score
- Computational efficiency
- Model robustness

These metrics help guide development and deployment decisions."""


def get_random_chunks(text, min_size=1, max_size=3):
    """Split text into random-sized chunks."""
    chunks = []
    remaining = text
    while remaining:
        # Random chunk size
        size = random.randint(min_size, max_size)
        chunk = remaining[:size]
        remaining = remaining[size:]
        chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    dashboard = TerminalDashboard(42)
    dashboard.start()

    try:
        batch = 0
        accumulated_text = ""

        # Get chunks of our test text
        chunks = get_random_chunks(TEST_TEXT)

        for i, chunk in enumerate(chunks):
            # Update various metrics
            train_loss = 1 / (i + 1) + random.uniform(0, 0.1)
            val_loss = train_loss + random.uniform(0, 0.05)

            # Accumulate text and update status
            accumulated_text += chunk
            dashboard.update_status(accumulated_text)

            # Update other dashboard elements
            dashboard.update_loss(train_loss)
            dashboard.update_batch(i)
            dashboard.update_step(i)
            dashboard.update_rate(0.5)

            # Test the info panel with sample data including memory
            info_dict = {
                "device": "cuda:0",
                "ram": "45.6%",
                "vram": "65.8%",
                "lr": f"{0.001 * (1 - i/100):.4f}",
            }
            dashboard.update_info(info_dict)

            # Add some test logs
            dashboard.logger.info(f"Processing chunk {i}")

            # Simulate processing time
            time.sleep(0.1)  # Shorter delay for faster testing

    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    finally:
        dashboard.stop()
