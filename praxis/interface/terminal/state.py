"""Terminal state management and restoration."""

import sys
import blessed


class TerminalStateManager:
    """Manages terminal state saving and restoration."""

    def __init__(self):
        self.term = blessed.Terminal()
        self.saved_terminal_state = None
        self.terminal_restored = False
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def save_state(self):
        """Save current terminal settings."""
        try:
            import termios
            import tty

            if hasattr(sys.stdin, "fileno"):
                try:
                    self.saved_terminal_state = termios.tcgetattr(sys.stdin.fileno())
                except:
                    self.saved_terminal_state = None
        except ImportError:
            self.saved_terminal_state = None

    def restore_terminal(self):
        """Fully restore terminal to its original state."""
        if self.terminal_restored:
            return

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

        self.terminal_restored = True

    def restore_terminal_safe(self):
        """Safe wrapper for terminal restoration that can be called from signal handlers."""
        try:
            if not self.terminal_restored:
                self.restore_terminal()
        except:
            # Last resort - try to show cursor at least
            try:
                sys.stderr.write("\033[?25h")  # ANSI escape to show cursor
                sys.stderr.flush()
            except:
                pass
