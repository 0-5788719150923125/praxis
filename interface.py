import blessed
import asciichartpy
import time
from collections import deque
from threading import Thread, Lock
import textwrap
import wcwidth


class TerminalDashboard:
    def __init__(self, max_data_points=50):
        self.term = blessed.Terminal()
        self.max_data_points = max_data_points
        self.train_losses = deque(maxlen=max_data_points)
        self.val_losses = deque(maxlen=max_data_points)
        self.status_text = "Initializing..."
        self.placeholder_text = "This is a static placeholder."
        self.step = 0
        self.running = False
        self.lock = Lock()

    def start(self):
        self.running = True
        Thread(target=self._run_dashboard).start()

    def stop(self):
        self.running = False

    def update_losses(self, train_loss, val_loss):
        with self.lock:
            self.train_losses.append(train_loss) if train_loss else None
            self.val_losses.append(val_loss) if val_loss else None

    def update_step(self, step):
        with self.lock:
            self.step = step

    def update_status(self, status):
        with self.lock:
            self.status_text = status

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
        # Wrap the text, preserving whitespace
        wrapped = textwrap.wrap(
            text, width=width, replace_whitespace=False, drop_whitespace=False
        )
        # Ensure each line is exactly 'width' characters, truncating if necessary
        return [self._truncate_to_width(line, width).ljust(width) for line in wrapped]

    def _run_dashboard(self):
        with self.term.fullscreen(), self.term.cbreak(), self.term.hidden_cursor():
            while self.running:
                # Clear the screen
                print(self.term.home + self.term.clear)

                # Calculate layout
                height = self.term.height - 4  # Leave some space for borders
                width = self.term.width - 3  # Leave space for middle border
                half_height = height // 2
                half_width = width // 2
                right_width = width - half_width - 1

                # Draw top border
                print("╔" + "═" * half_width + "╦" + "═" * right_width + "╗")

                # Prepare chart data
                with self.lock:
                    train_chart = self._draw_chart(
                        self.train_losses, half_width, half_height - 1
                    )  # Reduce height by 1
                    val_chart = self._draw_chart(
                        self.val_losses, half_width, half_height - 1
                    )  # Reduce height by 1

                # Prepare wrapped text
                status_lines = self._wrap_text(self.status_text, right_width)
                placeholder_lines = self._wrap_text(self.placeholder_text, right_width)

                # Draw content
                for i in range(height):
                    left_content = " " * half_width  # Default to empty space
                    right_content = " " * right_width  # Default to empty space

                    if i == 0:
                        left_content = "Training Loss".ljust(half_width)
                        right_content = "Feed".ljust(right_width)
                    elif i == 1:
                        left_content = "─" * half_width
                        right_content = "─" * right_width
                    elif i < half_height - 1:  # Reduce by 1 to make room for new border
                        if i - 2 < len(train_chart):
                            left_content = self._truncate_to_width(
                                train_chart[i - 2], half_width
                            ).ljust(half_width)
                        if i - 2 < len(status_lines):
                            right_content = status_lines[i - 2]
                    elif i == half_height - 1:  # New middle border
                        left_content = "═" * (half_width)
                        right_content = "═" * (right_width)
                    elif i == half_height:
                        left_content = "Validation Loss".ljust(half_width)
                        right_content = "Proto".ljust(right_width)
                    elif i == half_height + 1:
                        left_content = "─" * half_width
                        right_content = "─" * right_width
                    elif i > half_height + 1:
                        chart_index = i - half_height - 2
                        if chart_index < len(val_chart):
                            left_content = self._truncate_to_width(
                                val_chart[chart_index], half_width
                            ).ljust(half_width)
                        if i - half_height - 2 < len(placeholder_lines):
                            right_content = placeholder_lines[i - half_height - 2]

                    print(f"║{left_content}║{right_content}║")

                # Draw bottom border
                print("╚" + "═" * half_width + "╩" + "═" * right_width + "╝")

                # Add some information at the bottom
                with self.lock:
                    train_loss = self.train_losses[-1] if self.train_losses else 0
                    val_loss = self.val_losses[-1] if self.val_losses else 0
                    print(
                        f"PRAXIS | Step: {self.step}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )

                time.sleep(0.1)  # Update every 0.1 seconds


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
