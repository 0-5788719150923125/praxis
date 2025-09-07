"""Printing progress bar callback for training visualization."""

from typing import Any, Optional


def create_printing_progress_bar(
    process_position: int = 0,
    leave: bool = True,
    use_dashboard: bool = False,
    refresh_rate: int = 1,
) -> Optional[Any]:
    """Factory function to create a printing progress bar.

    This dynamically creates a progress bar class that inherits from the
    appropriate base class (Lightning or generic).

    Args:
        process_position: Position for multi-process environments
        leave: Whether to leave the progress bar after completion
        use_dashboard: Whether a dashboard is being used
        refresh_rate: How often to refresh the progress bar

    Returns:
        Progress bar instance or None if dashboard is used
    """
    if use_dashboard:
        return None

    # Get the appropriate base class dynamically
    from praxis.trainers import get_progress_bar_base

    base_class = get_progress_bar_base()

    # Create the PrintingProgressBar class dynamically
    class PrintingProgressBar(base_class):
        """Progress bar that handles printing in both terminal and Jupyter environments."""

        def __init__(self):
            """Initialize the printing progress bar."""
            super().__init__(refresh_rate, process_position, leave)
            self._last_print_lines = 0
            self._is_jupyter = self._check_jupyter()
            self._output_div = None

        def _check_jupyter(self) -> bool:
            """Check if running in a Jupyter environment."""
            try:
                from IPython import get_ipython

                return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
            except Exception:
                return False

        def _get_active_progress_bar(self):
            """Get the currently active progress bar."""
            active_progress_bar = None

            # Check train progress bar
            if (
                hasattr(self, "_train_progress_bar")
                and self._train_progress_bar is not None
                and not self.train_progress_bar.disable
            ):
                active_progress_bar = self.train_progress_bar
            # Check validation progress bar
            elif (
                hasattr(self, "_val_progress_bar")
                and self._val_progress_bar is not None
                and not self.val_progress_bar.disable
            ):
                active_progress_bar = self.val_progress_bar
            # Check test progress bar
            elif (
                hasattr(self, "_test_progress_bar")
                and self._test_progress_bar is not None
                and not self.test_progress_bar.disable
            ):
                active_progress_bar = self.test_progress_bar
            # Check predict progress bar
            elif (
                hasattr(self, "_predict_progress_bar")
                and self._predict_progress_bar is not None
                and not self.predict_progress_bar.disable
            ):
                active_progress_bar = self.predict_progress_bar

            return active_progress_bar

        def _escape_html(self, text: str) -> str:
            """Escape HTML special characters while preserving whitespace."""
            import html

            # First escape special characters
            escaped = html.escape(str(text))
            # Replace newlines with <br> tags to preserve formatting
            escaped = escaped.replace("\n", "<br>")
            # Replace spaces with &nbsp; to preserve multiple spaces
            escaped = escaped.replace("  ", "&nbsp;&nbsp;")
            return escaped

        def print(self, *args: Any, sep: str = " ", **kwargs: Any) -> None:
            """Print output, handling both terminal and Jupyter environments."""
            active_progress_bar = self._get_active_progress_bar()
            if active_progress_bar is None:
                return

            message = sep.join(map(str, args))

            if self._is_jupyter:
                from IPython.display import HTML, display

                # Escape the message for HTML
                safe_message = self._escape_html(message)

                # Create a dedicated output area if it doesn't exist
                if self._output_div is None:
                    self._output_div = display(
                        HTML(
                            '<div id="custom-output" style="white-space: pre-wrap;"></div>'
                        ),
                        display_id=True,
                    )

                # Update the output area with the escaped message
                self._output_div.update(
                    HTML(
                        f'<div id="custom-output" style="white-space: pre-wrap;">{safe_message}</div>'
                    )
                )
            else:
                print(message)

        @property
        def train_progress_bar(self):
            """Get training progress bar if it exists."""
            return getattr(self, "_train_progress_bar", None)
        
        @train_progress_bar.setter
        def train_progress_bar(self, value):
            """Set training progress bar."""
            self._train_progress_bar = value

        @property
        def val_progress_bar(self):
            """Get validation progress bar if it exists."""
            return getattr(self, "_val_progress_bar", None)
        
        @val_progress_bar.setter
        def val_progress_bar(self, value):
            """Set validation progress bar."""
            self._val_progress_bar = value

        @property
        def test_progress_bar(self):
            """Get test progress bar if it exists."""
            return getattr(self, "_test_progress_bar", None)
        
        @test_progress_bar.setter
        def test_progress_bar(self, value):
            """Set test progress bar."""
            self._test_progress_bar = value

        @property
        def predict_progress_bar(self):
            """Get prediction progress bar if it exists."""
            return getattr(self, "_predict_progress_bar", None)
        
        @predict_progress_bar.setter
        def predict_progress_bar(self, value):
            """Set prediction progress bar."""
            self._predict_progress_bar = value

    # Create and return an instance
    return PrintingProgressBar()


# For backward compatibility, export a class that can be imported
class PrintingProgressBar:
    """Static class for backward compatibility.

    Use create_printing_progress_bar() factory function instead.
    """

    def __new__(cls, *args, **kwargs):
        """Create a new instance using the factory function."""
        return create_printing_progress_bar(
            process_position=kwargs.get("process_position", 0),
            leave=kwargs.get("leave", True),
            use_dashboard=kwargs.get("use_dashboard", False),
            refresh_rate=kwargs.get("refresh_rate", 1),
        )
