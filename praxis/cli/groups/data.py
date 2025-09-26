"""Data-related CLI arguments."""

from praxis import RL_POLICIES_REGISTRY


class DataGroup:
    """Data configuration arguments."""

    name = "data"

    @classmethod
    def add_arguments(cls, parser):
        """Add data arguments to the parser."""
        group = parser.add_argument_group(cls.name)

        group.add_argument(
            "--data-path",
            type=str,
            nargs="+",
            default=None,
            help="Paths to a directory of files to use as training data",
        )

        group.add_argument(
            "--pile",
            action="store_true",
            default=False,
            help="Train exclusively on the minipile challenge dataset",
        )

        group.add_argument(
            "--phi",
            action="store_true",
            default=False,
            help="Supplement training with a mix of expert data",
        )

        group.add_argument(
            "--no-source",
            action="store_true",
            default=False,
            help="Disable training on the model's own source code",
        )

        group.add_argument(
            "--rl-type",
            type=str,
            default=None,
            choices=RL_POLICIES_REGISTRY.keys(),
            help="Enable reinforcement learning with specified algorithm. "
            "Note: Current GRPO implementation uses static dataset rewards (not true RL). "
            "True RL with generation will be added in a future update.",
        )
