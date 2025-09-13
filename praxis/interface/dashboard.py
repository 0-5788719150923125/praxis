"""Core terminal dashboard implementation."""

import atexit
import logging
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

        # Set up logging
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)  # Capture WARNING and above
        handler = DashboardStreamHandler(self)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Specifically configure the datasets logger to use our handler
        datasets_logger = logging.getLogger("datasets")
        datasets_logger.setLevel(logging.WARNING)
        # Remove any existing handlers to avoid duplicate output
        datasets_logger.handlers = []
        datasets_logger.addHandler(handler)
        datasets_logger.propagate = False  # Don't propagate to root logger

        # Also handle datasets.iterable_dataset specifically
        datasets_iterable_logger = logging.getLogger("datasets.iterable_dataset")
        datasets_iterable_logger.setLevel(logging.WARNING)
        datasets_iterable_logger.handlers = []
        datasets_iterable_logger.addHandler(handler)
        datasets_iterable_logger.propagate = False

        # Configure Lightning loggers to use our handler
        lightning_logger = logging.getLogger("lightning")
        lightning_logger.setLevel(
            logging.INFO
        )  # Lightning uses INFO for checkpoint messages
        lightning_logger.handlers = []
        lightning_logger.addHandler(handler)
        lightning_logger.propagate = False

        # Handle lightning.pytorch specifically (newer versions)
        lightning_pytorch_logger = logging.getLogger("lightning.pytorch")
        lightning_pytorch_logger.setLevel(logging.INFO)
        lightning_pytorch_logger.handlers = []
        lightning_pytorch_logger.addHandler(handler)
        lightning_pytorch_logger.propagate = False

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
                if random.random() < 0.1:
                    self.state.sign = -1 * self.state.sign
                value = "+" if self.state.sign == 1 else "-"
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

    def _run_dashboard(self):
        """Main dashboard rendering loop."""
        try:
            with managed_terminal(self.term, self.terminal_manager):
                while self.running:
                    try:
                        # Check for inactivity
                        inactive_time = self.activity_monitor.check_inactivity()
                        if inactive_time:
                            # Just note it internally, don't break the display
                            pass

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
