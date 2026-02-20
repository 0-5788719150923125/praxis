"""Logging and output-related CLI arguments."""


class LoggingGroup:
    """Logging and output configuration arguments."""

    name = "logging"

    @classmethod
    def add_arguments(cls, parser):
        """Add logging arguments to the parser."""
        group = parser.add_argument_group(cls.name)

        group.add_argument(
            "--no-dashboard",
            action="store_true",
            default=False,
            help="Disable the terminal dashboard",
        )

        group.add_argument(
            "--quiet",
            action="store_true",
            default=False,
            help="Suppress text generation in the terminal",
        )

        group.add_argument(
            "--headless",
            action="store_true",
            default=False,
            help="Server mode: disables progress bar and terminal inference output, but keeps web streaming active",
        )

        group.add_argument(
            "--debug",
            action="store_true",
            default=False,
            help="Print debug logs to the terminal",
        )
