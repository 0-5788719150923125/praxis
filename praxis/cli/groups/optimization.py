"""Optimization-related CLI arguments."""

from praxis import LOSS_REGISTRY, STRATEGIES_REGISTRY
from praxis.optimizers import OPTIMIZER_PROFILES


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
