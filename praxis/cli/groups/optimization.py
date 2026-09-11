from praxis import registry

"""Optimization-related CLI arguments."""


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
            registry="optimizers",
            default="Lion",
            help="The optimizer profile to use",
        )

        group.add_argument(
            "--loss-func",
            type=str,
            registry="losses",
            default="cross_entropy",
            help="The loss function to use",
        )

        group.add_argument(
            "--strategy",
            type=str,
            registry="strategies",
            default="naive",
            help="The multitask objective strategy to use for loss combination",
        )

        group.add_argument(
            "--task-weights",
            type=str,
            default=None,
            registry="task_weights",
            help=(
                "Named per-task loss weighting strategy from "
                "the task_weights registry. Unset = identity (every task at 1.0). "
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

        # Optimizer wrappers: an ordered, stackable list of registry keys
        # (applied innermost-first), replacing the old --trac/--ortho/
        # --lookahead/--schedule-free booleans.
        group.add_argument(
            "--optimizer-wrappers",
            nargs="*",
            default=[],
            registry="wrappers",
            metavar="WRAPPER",
            help=(
                "Optimizer wrappers to stack, in order. Choices: "
                + ", ".join(sorted(registry.namespace("wrappers").keys()))
                + ". E.g. --optimizer-wrappers ortho gated_schedule_free"
            ),
        )

        group.add_argument(
            "--fixed-schedule",
            action="store_true",
            default=False,
            help="Use a fixed (constant) learning rate schedule",
        )

        group.add_argument(
            "--gradient-clip-val",
            type=float,
            default=10.0,
            help=(
                "Global gradient-norm clip threshold. Watch the optimizer "
                "card's Clip Rate to tell whether it's binding. Ignored by "
                "trainers that don't support clipping (mono_forward/pipeline)."
            ),
        )
