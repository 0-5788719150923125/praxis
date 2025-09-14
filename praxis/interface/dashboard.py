"""Core terminal dashboard implementation."""

import atexit
import logging
import math
import random
import shutil
import sys
import time
import warnings
from collections import deque
from datetime import datetime
from threading import Lock, Thread

import blessed

from .io import DashboardOutput, DashboardStreamHandler, LogCapture
from .rendering import ChartRenderer, FrameBuilder, PanelRenderer, TextUtils
from .rendering.differential import TerminalDifferentialRenderer
from .state import ActivityMonitor, DashboardRegistry, MetricsState
from .terminal import TerminalStateManager, managed_terminal
from .visualization import ForestFireAutomata
from .web import DashboardStreamer


class TerminalDashboard:
    """Main terminal dashboard class."""

    def __init__(
        self, seed, arg_hash="000000", max_data_points=1000, max_log_lines=200
    ):
        # Core components
        self.term = blessed.Terminal()
        self.state = MetricsState(max_data_points)
        self.terminal_manager = TerminalStateManager()
        self.activity_monitor = ActivityMonitor()

        # Rendering components
        self.text_utils = TextUtils()
        self.frame_builder = FrameBuilder()
        self.chart_renderer = ChartRenderer()
        self.panel_renderer = PanelRenderer()
        self.differential_renderer = TerminalDifferentialRenderer(self.term)

        # State initialization
        self.state.seed = seed
        self.state.arg_hash = arg_hash
        self.max_log_lines = max_log_lines
        self.log_buffer = deque(maxlen=max_log_lines)

        # Display state
        self.running = False
        self.lock = Lock()
        self.previous_size = self._get_terminal_size()
        self.previous_frame = None
        self.fullscreen_log_mode = False  # Flag for fullscreen log mode
        self.log_scroll_offset = 0  # Scroll position for fullscreen log view

        # Correlation animation state
        self.correlation_frame = 0  # Frame counter for animations
        self.correlation_phase = 0.0  # Phase for oscillations
        self.correlation_twisted = False  # Whether to show twisted/mixed polarity

        # I/O management
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.dashboard_output = DashboardOutput(self.original_stdout)
        self.log_capture = LogCapture(self)

        # Visual elements
        self.game_of_life = None

        # Error handling
        self.error_exit = False
        self.error_message = None

        # Set up logging - intercept ALL loggers
        handler = DashboardStreamHandler(self)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Configure root logger to capture everything
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)  # Capture INFO and above from all sources

        # Remove all existing handlers from root logger to avoid duplicates
        root_logger.handlers = []
        root_logger.addHandler(handler)

        # Store original logging class to restore later if needed
        self._original_logger_class = logging.getLoggerClass()

        # Create a custom logger class that automatically uses our handler
        class DashboardLogger(logging.Logger):
            def __init__(self, name, level=logging.NOTSET):
                super().__init__(name, level)
                # Ensure all loggers created use the dashboard handler
                if name != "":  # Don't modify root logger again
                    self.handlers = []
                    self.addHandler(handler)
                    self.propagate = False

        # Set our custom logger class as the default
        logging.setLoggerClass(DashboardLogger)

        # For already-created loggers, retrofit them with our handler
        for name in list(logging.Logger.manager.loggerDict.keys()):
            logger = logging.getLogger(name)
            if isinstance(logger, logging.Logger):  # Skip PlaceHolder objects
                logger.handlers = []
                logger.addHandler(handler)
                logger.propagate = False
                # Use INFO as the default level for all loggers
                logger.setLevel(logging.INFO)

        # Capture warnings
        warnings.showwarning = self.show_warning

        # Register cleanup
        atexit.register(self._cleanup)

        # Set up streaming
        self._setup_streaming()

    def _setup_streaming(self):
        """Set up dashboard streaming if available."""
        try:
            from .web.streamer import register_dashboard as web_register_dashboard

            self._streamer = DashboardStreamer(self)

            # Register with both registry systems
            # 1. Register with DashboardRegistry (for internal use)
            registry = DashboardRegistry()
            registry.register("main", self)

            # 2. Register with web streamer (for API access)
            web_register_dashboard("main", self)

            if registry.socketio:
                self._streamer.start()
        except Exception:
            # Silently fail - don't break the dashboard
            pass

    def show_warning(self, message, category, filename, lineno, file=None, line=None):
        warning_message = warnings.formatwarning(
            message, category, filename, lineno, line
        )
        self.add_log(warning_message)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - always restore terminal."""
        self.stop()
        if not self.terminal_manager.terminal_restored:
            self.terminal_manager.restore_terminal()
        # Don't suppress exceptions
        return False

    def _cleanup(self):
        """Cleanup on exit."""
        if self.running:
            self.stop()

        # Restore original logger class
        if hasattr(self, "_original_logger_class"):
            logging.setLoggerClass(self._original_logger_class)

        # Always try to restore terminal on cleanup
        if not self.terminal_manager.terminal_restored:
            self.terminal_manager.restore_terminal()

    def _get_terminal_size(self):
        return shutil.get_terminal_size()

    def start(self):
        """Start the dashboard."""
        self.running = True

        # Save terminal state
        self.terminal_manager.save_state()

        # Register terminal restoration with shutdown manager
        try:
            from praxis.utils.system import get_shutdown_manager

            shutdown_manager = get_shutdown_manager()
            shutdown_manager.register_cleanup(
                self.terminal_manager.restore_terminal_safe, priority=10
            )
        except ImportError:
            pass

        # Capture stdout and stderr
        sys.stdout = self.log_capture
        sys.stderr = self.log_capture

        # Force unbuffered output
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(line_buffering=True)

        # Start dashboard thread
        Thread(target=self._run_dashboard).start()

    def stop(self, error=False):
        """Stop the dashboard."""
        self.running = False
        self.error_exit = error

        # Restore stdout/stderr first
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Perform terminal restoration
        if not self.terminal_manager.terminal_restored:
            self.terminal_manager.restore_terminal()

    def crash_with_error(self, error_text):
        """Immediately crash the dashboard and display the error."""
        # Restore original stdout/stderr first
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Force stop the dashboard
        self.running = False
        self.error_exit = True

        # Restore terminal
        if not self.terminal_manager.terminal_restored:
            self.terminal_manager.restore_terminal()

        # Small delay to ensure terminal mode switch completes
        time.sleep(0.1)

        # Show error
        print("\n\n", file=sys.stderr)
        print(
            "🚨 TRAINING CRASHED - Dashboard terminated to show error:", file=sys.stderr
        )
        print("=" * 80, file=sys.stderr)

        if not error_text or error_text.strip() == "":
            error_text = "Unknown error occurred (no error message captured)"

        print(error_text, file=sys.stderr)
        print("=" * 80, file=sys.stderr)

        self.error_message = error_text
        sys.stderr.flush()
        sys.exit(1)

    def add_log(self, message):
        """Add a log message."""
        with self.lock:
            # Strip ANSI codes and split the message into lines
            stripped_message = self.text_utils.strip_ansi(message)
            # Don't strip lines - preserve them as-is, but filter empty ones
            new_lines = [line for line in stripped_message.splitlines() if line]
            # Handle single line messages without newlines
            if not new_lines and stripped_message.strip():
                new_lines = [stripped_message.strip()]
            self.log_buffer.extend(new_lines)

    def force_redraw(self):
        """Force a full redraw."""
        self.previous_frame = None

    # Metric update methods - delegate to state
    def update_seed(self, seed):
        self.state.update_seed(seed)

    def set_mode(self, mode="train"):
        self.state.set_mode(mode)
        self.force_redraw()

    def set_start_time(self, time):
        self.state.set_start_time(time)

    def update_status(self, status):
        self.state.update_status(status)

    def update_params(self, total_params):
        self.state.update_params(total_params)
        self.force_redraw()

    def update_loss(self, train_loss):
        self.activity_monitor.mark_activity()
        self.state.update_loss(train_loss)

    def update_val(self, val_loss):
        self.state.update_val(val_loss)

    def update_accuracy(self, acc0, acc1):
        self.state.update_accuracy(acc0, acc1)

    def update_fitness(self, fitness):
        self.state.update_fitness(fitness)

    def update_memory(self, churn):
        self.state.update_memory(churn)

    def update_step(self, step):
        self.activity_monitor.mark_activity()
        self.state.update_step(step)

    def update_batch(self, batch):
        self.activity_monitor.mark_activity()
        self.state.update_batch(batch)

    def update_rate(self, seconds):
        self.state.update_rate(seconds)

    def update_tokens(self, num_tokens):
        self.state.update_tokens(num_tokens)

    def update_context_tokens(self, context_tokens):
        self.state.update_context_tokens(context_tokens)

    def update_expert_count(self, num_local, num_remote):
        self.state.update_expert_count(num_local, num_remote)

    def update_url(self, url):
        self.state.update_url(url)

    def update_info(self, info_dict):
        self.state.update_info(info_dict)

    def hours_since(self):
        return self.state.hours_since()

    def _update_screen(self, new_frame):
        """Update the terminal screen with a new frame."""
        # Correct borders for all lines
        new_frame = self.frame_builder.correct_borders(new_frame)

        # Use differential renderer for efficient character-level updates
        self.differential_renderer.render_frame(new_frame, self.dashboard_output)

        # Store frame for web streaming
        self.previous_frame = new_frame

    def _create_frame(self):
        """Create a new dashboard frame."""
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
        right_width = width - half_width

        frame = []
        frame.append(self.frame_builder.create_top_border(half_width, right_width))

        # Split the lower-left panel into two parts
        lower_left_quarter_width = half_width // 2
        lower_right_quarter_width = half_width - lower_left_quarter_width

        with self.lock:
            train_chart = self.chart_renderer.draw_chart(
                self.state.train_losses, right_width, (height // 2) - 1
            )
            sim_lines, self.game_of_life = self.panel_renderer.draw_simulation_panel(
                self.game_of_life, lower_left_quarter_width, (height // 2) - 1
            )
            info_chart = self.panel_renderer.draw_info_panel(
                self.state.info_dict, lower_right_quarter_width, (height // 2) - 1
            )

        # Wrap the entire status text
        status_lines = self.text_utils.wrap_text(self.state.status_text, half_width)

        # Calculate the maximum number of lines that can fit in the status section
        max_status_lines = (height // 2) - 3

        # If status_lines exceed max_status_lines, keep only the most recent lines
        if len(status_lines) > max_status_lines:
            status_lines = status_lines[-max_status_lines:]

        # Calculate available lines for LOG section
        log_available_lines = height - (height // 2) - 2 - 3
        log_text = "\n".join(list(self.log_buffer)[-log_available_lines:])
        log_lines = self.text_utils.wrap_text(log_text, right_width)

        # Pad status_lines and log_lines if they're shorter than the available space
        status_lines += [""] * (max_status_lines - len(status_lines))
        log_lines += [""] * (log_available_lines - len(log_lines))

        for i in range(height):
            left_content = " " * half_width
            right_content = " " * right_width

            if i == 0:
                # First line with metrics
                train_loss = (
                    self.state.train_losses[-1] if self.state.train_losses else 0
                )
                text = f" ERROR: {train_loss:.4f}"
                if self.state.val_loss is not None:
                    text += f" || VALIDATION: {self.state.val_loss:.4f}"
                if self.state.fitness is not None:
                    text += f" || FITNESS: {self.state.fitness:.4f}%"
                if self.state.memory_churn is not None:
                    text += f" || SURPRISE: {self.state.memory_churn:.2f}%"
                if self.state.accuracy is not None:
                    text += f" || ACCURACY: {self.state.accuracy[0]:.3f} || CONFIDENCE: {self.state.accuracy[1]:.3f}"
                right_content = self.text_utils.truncate_to_width(text, right_width)
                right_content = right_content.ljust(right_width)

                # Build the left content with HASH and CTX
                ctx_tokens = self.state.context_tokens
                left_text = f" HASH: {self.state.arg_hash} || CONTEXT: {ctx_tokens} {'token' if ctx_tokens == 1 else 'tokens'}"
                left_content = self.text_utils.truncate_to_width(left_text, half_width)
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
                # Update animation state
                self.correlation_frame += 1

                # Polarity changes - same as before (10% chance)
                if random.random() < 0.1:
                    self.state.sign = -1 * self.state.sign

                # Toggle twisted mode - 10% chance
                if random.random() < 0.1:
                    self.correlation_twisted = not self.correlation_twisted

                # Create oscillating patterns with varying frequencies
                # Primary wave - slow expansion/contraction
                primary_wave = math.sin(self.correlation_frame * 0.05)

                # Secondary wave - faster oscillation that modulates the primary
                secondary_wave = math.sin(self.correlation_frame * 0.15) * 0.3

                # Tertiary wave - even faster, creates "breathing" effect
                tertiary_wave = math.sin(self.correlation_frame * 0.25) * 0.2

                # Combine waves with phase shifting for more complex patterns
                self.correlation_phase += 0.02 + (math.sin(self.correlation_frame * 0.01) * 0.01)
                phase_shift = math.sin(self.correlation_phase) * 0.5

                # Calculate final amplitude (1-10 symbols, weighted towards fewer)
                combined = primary_wave + secondary_wave + tertiary_wave + phase_shift
                # Normalize to 0-1 range
                normalized = (combined + 2.0) / 4.0
                # Use a power function to bias towards lower values
                # Squaring makes lower values more common
                weighted = normalized ** 1.5  # Power of 1.5 gives nice distribution
                # Map to 1-10 symbols
                num_symbols = 1 + int(weighted * 9)

                # Build the symbol string based on twisted mode
                if self.correlation_twisted:
                    # Twisted mode - show mixed polarity
                    # Calculate the boundary position between + and -
                    # Use a different oscillation for the boundary shift
                    boundary_wave = math.sin(self.correlation_frame * 0.08) * 0.5
                    boundary_wave += math.sin(self.correlation_frame * 0.03) * 0.3
                    # Normalize boundary position to 0-1
                    boundary_pos = (boundary_wave + 1.0) / 2.0

                    # Calculate how many symbols are positive vs negative
                    num_positive = int(boundary_pos * num_symbols + 0.5)
                    num_negative = num_symbols - num_positive

                    # Determine primary polarity based on state.sign
                    if self.state.sign == 1:
                        # Positive dominant
                        value = "+" * num_positive + "-" * num_negative
                    else:
                        # Negative dominant
                        value = "-" * num_positive + "+" * num_negative
                else:
                    # Regular mode - single polarity
                    symbol = "+" if self.state.sign == 1 else "-"
                    value = symbol * num_symbols

                # Split the left section into two parts
                attention_label = f" CORRELATION: {value}"
                info_label = " INFO"
                attention_content = attention_label.ljust(lower_left_quarter_width)[
                    :lower_left_quarter_width
                ]
                info_content = info_label.ljust(lower_right_quarter_width)[
                    :lower_right_quarter_width
                ]
                left_content = attention_content + "║" + info_content
                right_content = " LOG (Press 'L' to fullscreen)".ljust(right_width)[
                    :right_width
                ]

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
                if chart_index < len(sim_lines):
                    sim_content = sim_lines[chart_index]
                if chart_index < len(info_chart):
                    info_content = info_chart[chart_index]

                # Ensure both parts are exactly the right width
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

            # Truncate and pad content
            left_content = self.text_utils.truncate_to_width(left_content, half_width)
            left_content = self.text_utils.visual_ljust(left_content, half_width)

            right_content = self.text_utils.truncate_to_width(
                right_content, right_width
            )
            right_content = self.text_utils.visual_ljust(right_content, right_width)

            # Combine the content with borders
            frame.append(f"║{left_content}║{right_content}║")

        frame.append(
            self.frame_builder.create_footer_separator(half_width, right_width)
        )

        with self.lock:
            elapsed = self.hours_since()
            footer_text = (
                f" PRAXIS:{str(self.state.seed)} | {self.state.total_params} | MODE: {self.state.mode} | "
                f"AGE: {elapsed:.2f}h | TOKENS: {self.state.num_tokens:.2f}B | BATCH: {int(self.state.batch)}, STEP: {int(self.state.step)}, "
                f"RATE: {self.state.rate:.2f}s | {self.state.local_experts} local experts, "
                f"{self.state.remote_experts} remote | {self.state.url}"
            )
            # Truncate and pad the footer text to fit the width
            footer_text = self.text_utils.truncate_to_width(footer_text, width + 1)
            footer_text = footer_text.ljust(width + 1)
            frame.append("║" + footer_text + "║")

        # Add bottom border
        frame.append(self.frame_builder.create_bottom_border(width))

        return frame

    def _handle_keyboard_input(self, key):
        """Handle keyboard input for dashboard controls."""
        # Toggle fullscreen log mode with 'l' or 'L' key
        if key.lower() == "l":
            self.fullscreen_log_mode = not self.fullscreen_log_mode
            # Clear the screen when switching modes
            self.previous_frame = None
            # Reset scroll position when entering/exiting fullscreen
            if not self.fullscreen_log_mode:
                self.log_scroll_offset = 0
            # Add a notification to the log
            mode_name = "fullscreen log" if self.fullscreen_log_mode else "dashboard"
            self.add_log(f"Switched to {mode_name} mode (press 'L' to toggle)")

        # Handle scrolling in fullscreen log mode
        elif self.fullscreen_log_mode:
            total_logs = len(self.log_buffer)
            available_height = self._get_terminal_size().lines - 2
            max_scroll = max(0, total_logs - available_height)

            if key.name == "KEY_UP":
                # Scroll up (show older logs)
                self.log_scroll_offset = min(self.log_scroll_offset + 1, max_scroll)
                self.previous_frame = None  # Force redraw
            elif key.name == "KEY_DOWN":
                # Scroll down (show newer logs)
                self.log_scroll_offset = max(self.log_scroll_offset - 1, 0)
                self.previous_frame = None  # Force redraw
            elif key.name == "KEY_PGUP":
                # Page up (scroll by full page)
                self.log_scroll_offset = min(self.log_scroll_offset + available_height, max_scroll)
                self.previous_frame = None  # Force redraw
            elif key.name == "KEY_PGDOWN":
                # Page down (scroll by full page)
                self.log_scroll_offset = max(self.log_scroll_offset - available_height, 0)
                self.previous_frame = None  # Force redraw
            elif key.name == "KEY_HOME":
                # Jump to beginning (oldest logs)
                self.log_scroll_offset = max_scroll
                self.previous_frame = None  # Force redraw
            elif key.name == "KEY_END":
                # Jump to end (newest logs)
                self.log_scroll_offset = 0
                self.previous_frame = None  # Force redraw

    def _create_fullscreen_log_frame(self):
        """Create a fullscreen view of the log output with scrolling support."""
        width, height = self._get_terminal_size()
        frame = []

        # Create top border using the same style as main dashboard
        # For fullscreen, we don't need the middle divider, so create a simple top border
        top_border = "╔" + "═" * (width - 2) + "╗"

        # Calculate scroll info
        total_logs = len(self.log_buffer)
        available_height = height - 2  # Subtract top and bottom border lines
        max_scroll = max(0, total_logs - available_height)

        # Create title with scroll position indicator
        if total_logs > available_height:
            # Show position in scrollable content
            current_position = total_logs - self.log_scroll_offset
            scroll_info = f" [{current_position}/{total_logs}] "
            title = f" LOG VIEW {scroll_info}(↑↓ scroll, PgUp/PgDn page, Home/End jump, L return) "
        else:
            title = " LOG VIEW (Press 'L' to return to dashboard) "

        title_start = (width - len(title)) // 2
        if title_start > 0 and title_start + len(title) < width - 1:
            top_border = (
                top_border[:title_start]
                + title
                + top_border[title_start + len(title) :]
            )
        frame.append(top_border)

        # Calculate content width (accounting for left and right borders)
        content_width = width - 4  # "║ " on left and " ║" on right

        # Get recent log entries (they're already in chronological order in the buffer)
        log_entries = list(self.log_buffer)

        # Apply scrolling offset
        if len(log_entries) > available_height:
            # Calculate which logs to show based on scroll offset
            # log_scroll_offset=0 means showing the most recent (bottom)
            # log_scroll_offset=max means showing the oldest (top)
            start_idx = len(log_entries) - available_height - self.log_scroll_offset
            end_idx = len(log_entries) - self.log_scroll_offset

            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(log_entries), end_idx)

            visible_logs = log_entries[start_idx:end_idx]

            # Pad if necessary (shouldn't happen with proper calculation)
            if len(visible_logs) < available_height:
                visible_logs = visible_logs + [""] * (available_height - len(visible_logs))
        else:
            # If we have fewer logs than screen space, pad at the top with empty lines
            visible_logs = [""] * (available_height - len(log_entries)) + log_entries

        # Add each log line to the frame with proper borders and padding
        for log_line in visible_logs:
            # Truncate or pad the log line to fit within content area
            if len(log_line) > content_width:
                formatted_line = log_line[: content_width - 3] + "..."
            else:
                formatted_line = log_line.ljust(content_width)

            # Add borders and 2-char buffer on the left
            frame.append(f"║ {formatted_line} ║")

        # Add bottom border
        bottom_border = "╚" + "═" * (width - 2) + "╝"
        frame.append(bottom_border)

        return frame

    def _run_dashboard(self):
        """Main dashboard rendering loop."""
        try:
            with managed_terminal(self.term, self.terminal_manager):
                while self.running:
                    try:
                        # Check for keyboard input (non-blocking)
                        key = self.term.inkey(timeout=0.01)
                        if key:
                            self._handle_keyboard_input(key)

                        # Check for inactivity
                        inactive_time = self.activity_monitor.check_inactivity()
                        if inactive_time:
                            # Just note it internally, don't break the display
                            pass

                        # Render either fullscreen log or normal dashboard
                        if self.fullscreen_log_mode:
                            new_frame = self._create_fullscreen_log_frame()
                        else:
                            new_frame = self._create_frame()

                        if not self.frame_builder.check_border_alignment(new_frame):
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
