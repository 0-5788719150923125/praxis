"""The annotated config behind the web app's Download button."""

import argparse

import pytest
import yaml

from praxis.cli.annotated_config import (
    _plain,
    reference_parser,
    render_annotated_config,
)
from praxis.cli.core import create_base_parser
from praxis.registry import Entry, Namespace


class _Documented:
    """A documented entry.

    Second paragraph, left out of the one-line description."""


class _Bare:
    pass


_REGISTRY = Namespace(
    "toy-kinds",
    {
        "documented": _Documented,
        "bare": _Bare,
        "profile": Entry(dict(a=1), "A profile with its own doc."),
    },
    doc="The toy namespace's doc, the long form of every flag bound to it.",
)


def _toy_parser():
    parser = create_base_parser()
    group = parser.add_argument_group("toy")
    group.add_argument(
        "--kind",
        registry=_REGISTRY,
        default="bare",
        help="which kind to use",
    )
    group.add_argument("--kinds", nargs="*", registry=_REGISTRY, default=[])
    group.add_argument(
        "--width",
        type=int,
        default=8,
        help="How wide",
        doc="The long form of --width, for the config file only.",
    )
    group.add_argument("--port", type=int, default=1, help="Port", exclude_hash=True)
    return parser


def _split(text):
    return text.split("EVERYTHING ELSE, AT ITS DEFAULT", 1)


def test_round_trips_to_the_rendered_config():
    config = {
        "kind": "documented",
        "kinds": ["profile", "bare"],
        "unknown_key": {"nested": [1, "two"]},
        "port": 5,
    }
    text = render_annotated_config(config, _toy_parser(), "toy")
    assert yaml.safe_load(text) == config


def test_used_keys_come_first_and_the_rest_is_commented_out():
    text = render_annotated_config({"kind": "documented"}, _toy_parser(), "toy")
    top, bottom = _split(text)
    assert "\nkind: documented  # default: bare\n" in top
    assert "\n# width: 8\n" in bottom
    assert "width" not in top
    assert all(not line or line.startswith("#") for line in bottom.splitlines())


def test_hash_excluded_flags_stay_out_of_the_reference_section():
    _, bottom = _split(render_annotated_config({}, _toy_parser(), "toy"))
    assert "port" not in bottom


def test_used_key_carries_help_doc_and_value_description():
    top, _ = _split(
        render_annotated_config(
            {"kind": "documented", "kinds": ["profile"]}, _toy_parser(), "toy"
        )
    )
    assert "# Which kind to use.\n" in top
    assert "# The toy namespace's doc, the long form of every flag bound to it." in top
    assert "#   documented\n#     A documented entry.\n" in top
    assert "#   profile\n#     A profile with its own doc.\n" in top
    assert "Second paragraph" not in top
    assert "\nkinds: [profile]  # default: []\n" in top


def test_default_note_only_when_the_value_differs():
    top, _ = _split(render_annotated_config({"width": 8}, _toy_parser(), "toy"))
    assert "\nwidth: 8\n" in top
    assert "# The long form of --width, for the config file only." in top


def test_every_real_flag_default_renders_and_round_trips():
    parser = reference_parser()
    config = {
        a.dest: _plain(a.default)
        for a in parser._actions
        if a.option_strings
        and not isinstance(a, argparse._HelpAction)
        and a.default is not argparse.SUPPRESS
    }
    text = render_annotated_config(config, parser, "everything")
    assert yaml.safe_load(text) == config
