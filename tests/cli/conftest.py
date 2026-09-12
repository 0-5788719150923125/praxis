"""A toy registry namespace and a parser with one flag of each kind bound to it."""

import random

import pytest

from praxis.cli.core import create_base_parser
from praxis.registry import Entry, Namespace


class _Documented:
    """A documented entry.

    Second paragraph, left out of the one-line description."""


class _Bare:
    pass


_TOY = Namespace(
    "toy-kinds",
    {
        "documented": _Documented,
        "bare": _Bare,
        "profile": Entry(dict(a=1), "A profile with its own doc."),
    },
    doc="The toy namespace's doc, the long form of every flag bound to it.",
)


@pytest.fixture
def toy_registry():
    return _TOY


@pytest.fixture
def toy_parser():
    parser = create_base_parser()
    group = parser.add_argument_group("toy")
    group.add_argument(
        "--kind",
        registry=_TOY,
        default="bare",
        help="which kind to use",
    )
    group.add_argument("--kinds", nargs="*", registry=_TOY, default=[])
    group.add_argument(
        "--width",
        type=int,
        default=8,
        help="How wide",
        doc="The long form of --width, for the config file only.",
    )
    group.add_argument("--port", type=int, default=1, help="Port", exclude_hash=True)
    group.add_argument(
        "--echo",
        type=int,
        default=0,
        help="The one sentence both forms open with. And a second one.",
        doc=("The one sentence both forms open with. Only the long form says " "this."),
    )
    group.add_argument(
        "--listy",
        nargs="*",
        default=[],
        help=(
            "Things to list. Space-separated (e.g. '--listy a b'). "
            "Available: a, b, c."
        ),
    )
    group.add_argument(
        "--rolled",
        type=int,
        default=random.randrange(1000),
        volatile_default=True,
        help="A value drawn fresh on each run",
    )
    return parser
