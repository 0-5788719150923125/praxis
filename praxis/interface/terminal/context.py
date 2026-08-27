"""Terminal context management."""

from contextlib import contextmanager

import blessed


@contextmanager
def managed_terminal(term, state_manager):
    """Hold the alternate screen for the lifetime of the render loop.

    ``cbreak``/``hidden_cursor`` stay with blessed, but the alternate screen
    belongs to the state manager so that entering and leaving it happen exactly
    once - see TerminalStateManager. The screen is released here, on the render
    thread, precisely because the thread that paints must be the one that
    stops painting before anyone else writes to the terminal.
    """
    state_manager.enter_fullscreen()
    try:
        with term.cbreak(), term.hidden_cursor():
            yield
    finally:
        state_manager.restore_terminal()
