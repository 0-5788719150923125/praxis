"""Other/miscellaneous CLI arguments."""

import math
import random


class OtherGroup:
    """Miscellaneous configuration arguments."""

    name = "other"

    @classmethod
    def add_arguments(cls, parser):
        """Add other arguments to the parser."""
        group = parser.add_argument_group(cls.name)

        group.add_argument(
            "--seed",
            type=int,
            default=int(65536 * (2 * math.acos(1 - random.random()) / math.pi) ** 6.66),
            help="Global seed used for reproducibility",
        )

        group.add_argument(
            "--meta",
            type=str,
            action="append",
            default=[],
            help="Append keywords to a list at 'config.meta'. Used for model development. You probably don't need this.",
        )

        group.add_argument(
            "--eval-every",
            type=int,
            default=None,
            help="Run partial evaluation every N validation intervals",
        )

        group.add_argument(
            "--eval-tasks",
            type=str,
            default="helm|hellaswag|2|1,lighteval|glue:cola|2|1,lighteval|coqa|2|1",
            help="Run a subset of evaluation tests after each validation step. This can be slow.",
        )

        group.add_argument(
            "--reset",
            action="store_true",
            default=False,
            help="Reset the checkpoint",
        )

        group.add_argument(
            "--preserve",
            action="store_true",
            default=False,
            help="Mark this run as preserved (protected from --reset)",
        )

        group.add_argument(
            "--list-runs",
            action="store_true",
            default=False,
            help="List all available runs and exit",
        )

    @classmethod
    def add_dev_argument_if_needed(cls, parser):
        """Add --dev argument if it wasn't already added by environment loading."""
        # Check if --dev already exists
        for action in parser._actions:
            if "--dev" in action.option_strings:
                return

        # Find the 'other' group
        for group in parser._action_groups:
            if group.title == "other":
                group.add_argument(
                    "--dev",
                    action="store_true",
                    default=False,
                    help="Use fewer resources (3 layers, smaller datasets, etc), always start from a new model (i.e. force '--reset'), and never conflict/remove existing, saved models. Can be used simultaneously alongside an active, running 'live' model.",
                )
                break
