"""Terminal state management and restoration."""

import sys
from threading import RLock

import blessed


class TerminalStateManager:
    """Manages terminal state saving and restoration.

    This object is the single owner of the alternate screen. It used to share
    that job with ``blessed``'s ``term.fullscreen()`` context, which meant the
    exit sequence was written twice on shutdown - once by the context manager
    and once here. A second ``rmcup`` on the normal buffer is not a no-op: it
    restores the cursor position saved back when ``smcup`` ran, so subsequent
    output lands on top of the user's scrollback. Enter and exit both go
    through this class now, exactly once each.
    """

    def __init__(self):
        self.term = blessed.Terminal()
        self.saved_terminal_state = None
        self.terminal_restored = False
        self.in_fullscreen = False
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        # Restoration can be reached from the render thread, the signal
        # handler's cleanup thread, atexit, and the trainer's own teardown.
        self._lock = RLock()

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

    def reset(self):
        """Begin a fresh terminal session (the dashboard can be restarted)."""
        with self._lock:
            self.terminal_restored = False
            self.in_fullscreen = False

    def enter_fullscreen(self):
        """Switch to the alternate screen buffer."""
        with self._lock:
            if self.in_fullscreen or self.terminal_restored:
                return
            try:
                print(self.term.enter_fullscreen, end="", file=self.original_stderr)
                self.original_stderr.flush()
                self.in_fullscreen = True
            except Exception:
                pass

    def restore_terminal(self):
        """Fully restore terminal to its original state."""
        with self._lock:
            if self.terminal_restored:
                return
            # Claim it first: a second caller arriving mid-restore must not
            # start writing escape sequences of its own.
            self.terminal_restored = True

            try:
                if hasattr(self, "original_stderr"):
                    # Leave the alternate screen, but only if we entered it -
                    # an unmatched rmcup moves the cursor into the scrollback.
                    if self.in_fullscreen:
                        print(
                            self.term.exit_fullscreen,
                            end="",
                            file=self.original_stderr,
                        )
                        self.in_fullscreen = False
                    # Reset all terminal attributes
                    print(self.term.normal, end="", file=self.original_stderr)
                    # Make cursor visible. `visible_cursor` is an empty string
                    # on some blessed/terminfo combinations, which made this a
                    # silent no-op - the cursor only came back because blessed's
                    # hidden_cursor context happened to emit cnorm on its way
                    # out, and that context is exactly what an emergency restore
                    # never reaches. Fall back to the raw sequence.
                    print(
                        self.term.normal_cursor
                        or self.term.visible_cursor
                        or "\033[?25h",
                        end="",
                        file=self.original_stderr,
                    )
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
