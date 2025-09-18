"""Training-related CLI arguments."""

from praxis.trainers import TRAINER_REGISTRY


class TrainingGroup:
    """Training configuration arguments."""

    name = "training"

    @classmethod
    def add_arguments(cls, parser):
        """Add training arguments to the parser."""
        group = parser.add_argument_group(cls.name)

        group.add_argument(
            "--trainer-type",
            type=str,
            choices=list(TRAINER_REGISTRY.keys()),
            default="backpropagation",
            help="Training strategy to use (backpropagation: standard gradient descent, mono_forward: process-based pipeline parallelism with O(1) memory)",
        )

        # Mono-Forward specific arguments
        group.add_argument(
            "--mono-forward-prediction-mode",
            type=str,
            choices=["ff", "bp"],
            default="bp",
            help="Mono-Forward prediction mode: 'ff' sums all layer goodness scores, 'bp' uses only last layer (default: bp)",
        )

        group.add_argument(
            "--mono-forward-vocab-reduction",
            type=int,
            default=4,
            help="Factor to reduce vocabulary size for internal Mono-Forward layers (default: 4)",
        )

        group.add_argument(
            "--max-steps",
            type=int,
            default=None,
            help="Maximum number of training steps (None for infinite training)",
        )

        group.add_argument(
            "--pipeline-depth",
            type=int,
            default=4,
            help="Number of batches to keep in pipeline for mono_forward trainer",
        )