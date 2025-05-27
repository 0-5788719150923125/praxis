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
        self.error_buffer = []
        self.in_traceback = False

    def write(self, data):
        if "Traceback (most recent call last):" in data:
            self.in_traceback = True
            self.error_buffer = [data]
            return

        if self.in_traceback:
            self.error_buffer.append(data)
            if data.strip() and not data.startswith(" "):
                self.in_traceback = False
                full_error = "".join(self.error_buffer)
                # Store error in dashboard for later
                self.dashboard.error_message = full_error
                self.dashboard.stop(error=True)
                return

        self.log_buffer.write(data)
        self.dashboard.add_log(data.rstrip())

    def flush(self):
        self.log_buffer.flush()


class TerminalDashboard:
    def __init__(self, seed, max_data_points=1000, max_log_lines=200):
        self.seed = seed
        self.term = blessed.Terminal()
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        self.max_data_points = max_data_points
        self.train_losses = deque(maxlen=max_data_points)
        self.sign = 1
        self.val_loss = None
        self.status_text = "_initializing"
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
        self.stop()
        # if not self.error_exit:
        #     self._reset_terminal()

    def _reset_terminal(self):
        print(
            self.term.normal
            + self.term.clear
            + self.term.home
            + self.term.visible_cursor,
            end="",
        )
        sys.stdout.flush()

    def _get_terminal_size(self):
        return shutil.get_terminal_size()

    @contextmanager
    def managed_terminal(self):
        try:
            with self.term.fullscreen(), self.term.cbreak(), self.term.hidden_cursor():
                yield
        finally:
            pass
            # if not self.error_exit:  # Only reset if not error exit
            #     self._reset_terminal()

    def start(self):
        self.running = True
        sys.stdout = self.log_capture
        sys.stderr = self.log_capture
        Thread(target=self._run_dashboard).start()

    def stop(self, error=False):
        self.running = False
        self.error_exit = error
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

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

    def update_params(self, total_params):
        with self.lock:
            reduced = int(total_params / 10**6)
            self.total_params = f"{reduced}M"
            self.previous_frame = None  # force a redraw

    def update_loss(self, train_loss):
        with self.lock:
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
            self.step = step

    def update_batch(self, batch):
        with self.lock:
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

        with self.lock:
            train_chart = self._draw_chart(
                self.train_losses, right_width, (height // 2) - 1
            )
            sim_chart = self._draw_simulation(half_width, (height // 2) - 1)

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
                left_content = self._truncate_to_width(f" HOST", half_width)
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
                left_content = f" ATTENTION: {self.sign:+.1f}"
                left_content = left_content.ljust(half_width)[:half_width]
                right_content = " LOG".ljust(right_width)[:right_width]
            elif i == (height // 2) + 1:
                left_content = "─" * half_width
                right_content = "─" * right_width
            elif i > (height // 2) + 1:
                chart_index = i - (height // 2) - 2
                if chart_index < len(sim_chart):
                    left_content = sim_chart[chart_index]
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

    def _run_dashboard(self):
        try:
            with self.managed_terminal():
                while self.running:
                    try:
                        new_frame = self._create_frame()
                        if not self._check_border_alignment(new_frame):
                            self.previous_frame = None
                        self._update_screen(new_frame)
                        time.sleep(0.1)
                    except Exception as e:
                        self.add_log(f"Dashboard error: {str(e)}")
                        time.sleep(1)
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

            # Add some test logs
            dashboard.logger.info(f"Processing chunk {i}")

            # Simulate processing time
            time.sleep(0.1)  # Shorter delay for faster testing

    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    finally:
        dashboard.stop()
