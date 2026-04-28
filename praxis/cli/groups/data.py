"""Data-related CLI arguments."""

from praxis import RL_POLICIES_REGISTRY, SAMPLER_REGISTRY
from praxis.data.config import DATASET_COLLECTIONS


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
            action="extend",
            default=None,
            help="Paths to directories of files to use as training data (can be specified multiple times)",
        )

        group.add_argument(
            "--train-datasets",
            type=str,
            nargs="+",
            default=["base"],
            help=(
                "Named dataset collections to train on. Space- or comma-separated "
                "(e.g. '--train-datasets base phi' or '--train-datasets base,phi'). "
                f"Available: {', '.join(sorted(DATASET_COLLECTIONS.keys()))}."
            ),
        )

        group.add_argument(
            "--validation-datasets",
            type=str,
            nargs="+",
            default=["validation"],
            help=(
                "Named dataset collections to use for validation. Same format "
                "as --train-datasets."
            ),
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

        group.add_argument(
            "--sampler-mode",
            type=str,
            default="loss",
            choices=SAMPLER_REGISTRY.keys(),
            help="Dataset sampling weighting mode: 'loss' (upsample high-loss datasets via per-sequence CE), "
            "'novelty' (bigram novelty via Count-Min Sketch), "
            "'dynamic' (EMA of token counts), 'static' (use weights from DATASET_COLLECTIONS as-is), "
            "or 'uniform' (force every dataset to 1.0, ignoring DATASET_COLLECTIONS).",
        )
