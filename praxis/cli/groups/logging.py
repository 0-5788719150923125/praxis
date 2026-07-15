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
            "--infer-every",
            type=int,
            default=3,
            help="Seconds between inference token generation during training (default: 3)",
        )

        group.add_argument(
            "--infer-context",
            type=int,
            default=None,
            help="Max context length (in tokens/bytes) for inference during training",
        )

        group.add_argument(
            "--debug",
            action="store_true",
            default=False,
            help="Print debug logs to the terminal",
        )

        group.add_argument(
            "--profile-memory",
            action="store_true",
            default=False,
            help="Record a CUDA memory snapshot over a step window (load at https://pytorch.org/memory_viz)",
        )

        group.add_argument(
            "--profile-memory-start",
            type=int,
            default=0,
            help="Global step to begin memory recording, to skip warmup "
            "(default: 0). The first batch of the process is always skipped "
            "so torch.compile is never recorded",
        )

        group.add_argument(
            "--profile-memory-steps",
            type=int,
            default=50,
            help="Number of steps to record before dumping the snapshot (default: 50)",
        )

        group.add_argument(
            "--profile-memory-max-entries",
            type=int,
            default=5_000_000,
            help="Max allocation events retained in the trace ring buffer. Larger spans more steps but grows the pickle (default: 5M ~ 150 steps)",
        )
