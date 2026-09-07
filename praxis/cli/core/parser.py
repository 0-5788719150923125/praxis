"""Parser creation, argument registration metadata, and custom help formatting."""

import argparse

from .hasher import register_hash_exclusion


class _HashAwareContainer:
    """Mixin that teaches ``add_argument`` about Praxis registration metadata.

    ``exclude_hash=True`` is why this exists. A run's identity is a hash over
    argv, and it names the run's checkpoint directory - so a flag that changes
    nothing about the model (where it is served, how loud it logs, which decoder
    it uses at inference) must be kept out of it, or toggling it forks a new
    empty run. That used to be a hand-kept list of flag strings in hasher.py,
    sitting a long way from the arguments it named and free to drift from them.
    Declared here, the fact lives on the argument itself and cannot go stale.

    Anything else that wants to be said about an argument at its definition site
    belongs here too - add a keyword, consume it, pass the rest through.
    """

    def add_argument(self, *args, exclude_hash=False, **kwargs):
        action = super().add_argument(*args, **kwargs)
        if exclude_hash:
            if not action.option_strings:
                # The hasher keys positionals by index (_pos_N), not by name,
                # so there is no flag string to exclude. Fail loudly rather
                # than accept a declaration that would quietly do nothing.
                raise ValueError(
                    f"exclude_hash is only meaningful for optional arguments; "
                    f"{action.dest!r} is positional."
                )
            register_hash_exclusion(*action.option_strings)
        return action

    # Nested containers have to stay hash-aware, or an argument added through
    # one silently loses the keyword. Integrations reach the groups by walking
    # parser._action_groups, so these are the objects they end up calling.
    def add_argument_group(self, *args, **kwargs):
        group = PraxisArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        return group

    def add_mutually_exclusive_group(self, **kwargs):
        group = PraxisMutuallyExclusiveGroup(self, **kwargs)
        self._mutually_exclusive_groups.append(group)
        return group


class PraxisArgumentGroup(_HashAwareContainer, argparse._ArgumentGroup):
    """An argument group whose ``add_argument`` accepts ``exclude_hash``."""


class PraxisMutuallyExclusiveGroup(
    _HashAwareContainer, argparse._MutuallyExclusiveGroup
):
    """A mutually-exclusive group whose ``add_argument`` accepts ``exclude_hash``."""


class PraxisArgumentParser(_HashAwareContainer, argparse.ArgumentParser):
    """The Praxis parser: argparse plus per-argument registration metadata.

    Every group handed out by this parser is a :class:`PraxisArgumentGroup`,
    including the ones argparse builds for itself, so ``exclude_hash`` works
    whether an argument is added to the parser, to a group a CLI group class
    created, or to a group an integration found by title.
    """


def wrap_green(text):
    """Wrap text in ANSI green color codes."""
    return f"\033[92m{text}\033[00m"


class CustomHelpFormatter(argparse.HelpFormatter):
    """Custom help formatter with better formatting and type information."""

    def __init__(self, prog, indent_increment=2, max_help_position=30, width=None):
        # Use terminal width if available, otherwise use a sensible default
        if width is None:
            try:
                import shutil

                width = shutil.get_terminal_size().columns
            except (ImportError, AttributeError):
                width = 100

        # Adjust max_help_position based on terminal width
        max_help_position = min(30, width // 3)

        super().__init__(prog, indent_increment, max_help_position, width)

    def _format_usage(self, usage, actions, groups, prefix):
        return ""  # This effectively removes the usage section

    def _format_action_invocation(self, action):
        """Customizes how arguments are displayed in the help output."""
        if action.option_strings:
            # It's an optional argument
            if action.nargs == 0:
                # It's a flag (like --verbose)
                return ", ".join(action.option_strings)
            else:
                # It takes a value (like --file <value>)
                return f"{', '.join(action.option_strings)} <value>"

    def _get_help_string(self, action):
        help_text = action.help or ""

        # Add type information when available
        if action.type is not None and hasattr(action.type, "__name__"):
            type_name = action.type.__name__
            if str(type_name) == "<lambda>":
                type_name = "str"
            help_text = f"({wrap_green(type_name)}) {help_text}"
        elif isinstance(action, argparse._StoreTrueAction) or isinstance(
            action, argparse._StoreFalseAction
        ):
            # It's a boolean flag
            help_text = f"({wrap_green('bool')}) {help_text}"

        # Add choices information when available (but only in the help text)
        if action.choices is not None:
            choice_str = ", ".join([str(c) for c in action.choices])
            help_text = f"{help_text} (choices: {choice_str})"

        # Add default value information when available
        if action.default is not argparse.SUPPRESS:
            # Always show default, even if it's None
            help_text = f"{help_text} (default: {str(action.default)})"

        return help_text


def create_base_parser(description="Praxis CLI"):
    """Create the base argument parser with custom formatting."""
    return PraxisArgumentParser(
        description=description,
        formatter_class=CustomHelpFormatter,
    )
