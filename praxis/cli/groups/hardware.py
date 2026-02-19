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

        group.add_argument(
            "--num-nodes",
            type=int,
            default=1,
            help="Number of nodes for distributed training (default: 1)",
        )

        group.add_argument(
            "--node-rank",
            type=int,
            default=None,
            help="Rank of this node among all nodes (overrides NODE_RANK env var)",
        )

        group.add_argument(
            "--master-addr",
            type=str,
            default=None,
            help="Hostname or IP of the rank-0 node (overrides MASTER_ADDR env var)",
        )

        group.add_argument(
            "--master-port",
            type=int,
            default=None,
            help="Port for distributed rendezvous on the rank-0 node (overrides MASTER_PORT env var)",
        )
