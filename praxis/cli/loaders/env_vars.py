"""Environment variable overrides for CLI arguments.

Every argparse argument registered on the parser is also settable via an
environment variable named `PRAXIS_<DEST>` (uppercased, underscores kept).
Explicit CLI flags always win; otherwise the env var overrides experiment
YAML defaults and the hard-coded argparse default.

Value coercion handles the common argparse action types:
    - store_true / store_false / store_const -> parsed as bool-ish string
    - regular store with `type=int`/`type=float`/etc. -> the declared type
    - `choices=[...]` -> validated against the choices list
    - `nargs` in {'+', '*', N>1} -> split on commas or parsed as JSON array
    - `action='append'`/`action='extend'` -> same list splitting
"""

import argparse
import os
from typing import Any, Optional

from praxis.utils import coerce_to_list

DEFAULT_PREFIX = "PRAXIS_"

_TRUE_STRINGS = {"1", "true", "yes", "on", "y", "t"}
_FALSE_STRINGS = {"0", "false", "no", "off", "n", "f", ""}


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in _TRUE_STRINGS:
        return True
    if lowered in _FALSE_STRINGS:
        return False
    raise ValueError(f"invalid boolean value '{raw}'")


def _coerce(action: argparse.Action, raw: str) -> Any:
    """Convert a raw env string to the value argparse would produce."""
    cls_name = action.__class__.__name__

    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return _parse_bool(raw)

    # store_const: truthy env triggers the const, falsy keeps the default.
    if cls_name == "_StoreConstAction":
        return action.const if _parse_bool(raw) else action.default

    type_fn = action.type or (lambda s: s)

    is_list = (
        action.nargs in ("+", "*")
        or (isinstance(action.nargs, int) and action.nargs > 1)
        or cls_name in ("_AppendAction", "_ExtendAction", "_AppendConstAction")
    )

    if is_list:
        parts = coerce_to_list(raw)
        converted = [type_fn(p) for p in parts]
        if action.choices is not None:
            for item in converted:
                if item not in action.choices:
                    raise ValueError(f"{item!r} not in choices {action.choices!r}")
        return converted

    converted = type_fn(raw)
    if action.choices is not None and converted not in action.choices:
        raise ValueError(f"{converted!r} not in choices {action.choices!r}")
    return converted


def _should_skip(action: argparse.Action) -> bool:
    """Actions that don't correspond to a user-facing value."""
    cls_name = action.__class__.__name__
    if cls_name in ("_HelpAction", "_VersionAction", "_SubParsersAction"):
        return True
    if action.dest in (argparse.SUPPRESS, None, ""):
        return True
    return False


class EnvVarLoader:
    """Apply `PRAXIS_*` environment variable overrides to parsed args."""

    def __init__(self, prefix: str = DEFAULT_PREFIX):
        self.prefix = prefix
        self.applied: dict = {}

    def env_name_for(self, dest: str) -> str:
        return f"{self.prefix}{dest.upper()}"

    def apply_env_vars(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        explicitly_provided: Optional[set] = None,
    ) -> argparse.Namespace:
        """Override `args` from environment variables.

        Any argument the user set explicitly on the CLI is left alone.
        Unknown or malformed env values emit a warning and are skipped.
        """
        if explicitly_provided is None:
            explicitly_provided = set()

        for action in parser._actions:
            if _should_skip(action):
                continue

            dest = action.dest
            cli_name = dest.replace("_", "-")
            if dest in explicitly_provided or cli_name in explicitly_provided:
                continue

            env_name = self.env_name_for(dest)
            if env_name not in os.environ:
                continue

            raw = os.environ[env_name]
            try:
                value = _coerce(action, raw)
            except ValueError as exc:
                print(
                    f"Warning: {env_name}={raw!r} could not be parsed "
                    f"for --{cli_name}: {exc}"
                )
                continue

            setattr(args, dest, value)
            self.applied[dest] = raw

        return args
