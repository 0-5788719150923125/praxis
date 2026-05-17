"""Optimization-related CLI arguments."""

from praxis import LOSS_REGISTRY, STRATEGIES_REGISTRY
from praxis.optimizers import OPTIMIZER_PROFILES
from praxis.tasks import TASK_WEIGHTER_REGISTRY


class OptimizationGroup:
    """Optimization configuration arguments."""

    name = "optimization"

    @classmethod
    def add_arguments(cls, parser):
        """Add optimization arguments to the parser."""
        group = parser.add_argument_group(cls.name)

        group.add_argument(
            "--optimizer",
            type=str,
            choices=OPTIMIZER_PROFILES.keys(),
            default="Lion",
            help="The optimizer profile to use",
        )

        group.add_argument(
            "--loss-func",
            type=str,
            choices=LOSS_REGISTRY.keys(),
            default="cross_entropy",
            help="The loss function to use",
        )

        group.add_argument(
            "--strategy",
            type=str,
            choices=STRATEGIES_REGISTRY.keys(),
            default="naive",
            help="The multitask objective strategy to use for loss combination",
        )

        group.add_argument(
            "--task-weights",
            type=str,
            default=None,
            choices=sorted(TASK_WEIGHTER_REGISTRY.keys()),
            help=(
                "Named per-task loss weighting strategy from "
                "TASK_WEIGHTER_REGISTRY. Unset = identity (every task at 1.0). "
                "Fixed variants use constant scalars; learnable variants use a "
                "sigmoid-gated per-task parameter with an L2 anchor."
            ),
        )

        group.add_argument(
            "--no-mask-prompts",
            action="store_true",
            default=False,
            help=(
                "Drop the assistant_mask before composing loss weights so "
                "every token contributes (the pre-2bc2cd4 language-modeling "
                "objective). Default off, meaning prompts are masked. "
                "Useful for small models that lack the capacity for the "
                "SFT-style prompt-conditional split."
            ),
        )

        # Optimizer wrappers
        group.add_argument(
            "--trac",
            action="store_true",
            default=False,
            help="Wrap the optimizer in TRAC, which can mitigate the loss of plasticity over time",
        )

        group.add_argument(
            "--ortho",
            action="store_true",
            default=False,
            help="Wrap the optimizer in OrthoGrad, projecting gradients to be orthogonal to parameters",
        )

        group.add_argument(
            "--lookahead",
            action="store_true",
            default=False,
            help="Wrap the optimizer in Lookahead",
        )

        group.add_argument(
            "--fixed-schedule",
            action="store_true",
            default=False,
            help="Use a fixed (constant) learning rate schedule",
        )

        group.add_argument(
            "--schedule-free",
            action="store_true",
            default=False,
            help="Use the Schedule-Free optimizer wrapper",
        )
