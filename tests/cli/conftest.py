"""A toy registry namespace and a parser with one flag of each kind bound to it."""

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
    return parser
