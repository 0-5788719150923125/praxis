"""Hardware-related CLI arguments."""


class HardwareGroup:
    """Hardware configuration arguments."""

    name = "hardware"

    @classmethod
    def add_arguments(cls, parser):
        """Add hardware arguments to the parser."""
        group = parser.add_argument_group(cls.name)

        group.add_argument(
            "--device",
            type=str,
            default="cpu",
            help="Device to use",
        )

        group.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help="Batch size to use for training",
        )

        group.add_argument(
            "--checkpoint-every",
            type=int,
            default=None,
            help="Apply gradient checkpointing every X layers",
        )
