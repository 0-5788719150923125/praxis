"""Terminal context management."""

from contextlib import contextmanager
import blessed


@contextmanager
def managed_terminal(term, state_manager):
    """Context manager for terminal fullscreen mode."""
    try:
        with term.fullscreen(), term.cbreak(), term.hidden_cursor():
            yield
    finally:
        # Ensure terminal is restored when exiting the context
        if not state_manager.terminal_restored and not getattr(
            state_manager, "error_exit", False
        ):
            state_manager.restore_terminal()
