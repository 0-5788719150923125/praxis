"""Hardware-related CLI arguments."""

from praxis import registry
from praxis.trainers.precision import DEFAULT_PRECISION, PRECISION_CHOICES


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
            "--precision",
            type=str,
            default=DEFAULT_PRECISION,
            choices=PRECISION_CHOICES,
            registry="precision",
            metavar="{" + ",".join(registry.namespace("precision")) + "}",
            help=(
                "Numeric precision for weights, gradients and matmul kernels. "
                "Common spellings (fp16, bf16, half, ...) resolve to these names"
            ),
        )

        group.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help="Batch size to use for training",
            doc=(
                "Rows per microbatch, the unit that has to fit in memory. "
                "--target-batch-size sets rows per optimizer step, reached by "
                "accumulating microbatches. Larger batches also unlock the longer "
                "sequence-length tiers (see --block-size)."
            ),
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
            help="Number of nodes for distributed training",
            exclude_hash=True,
        )

        group.add_argument(
            "--node-rank",
            type=int,
            default=None,
            help="Rank of this node among all nodes (overrides NODE_RANK env var)",
            exclude_hash=True,
        )

        group.add_argument(
            "--master-addr",
            type=str,
            default=None,
            help="Hostname or IP of the rank-0 node (overrides MASTER_ADDR env var)",
            exclude_hash=True,
        )

        group.add_argument(
            "--master-port",
            type=int,
            default=None,
            help="Port for distributed rendezvous on the rank-0 node (overrides MASTER_PORT env var)",
            exclude_hash=True,
        )
