"""Parser creation and custom help formatting."""

import argparse


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
    return argparse.ArgumentParser(
        description=description,
        formatter_class=CustomHelpFormatter,
    )
