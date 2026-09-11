"""Tests for praxis/cli/annotated_config.py, the annotated config behind the web
app's Download button."""

import argparse

import yaml

from praxis.cli.annotated_config import (
    _plain,
    reference_parser,
    render_annotated_config,
)


def _split(text):
    return text.split("EVERYTHING ELSE, AT ITS DEFAULT", 1)


def test_round_trips_to_the_rendered_config(toy_parser):
    config = {
        "kind": "documented",
        "kinds": ["profile", "bare"],
        "unknown_key": {"nested": [1, "two"]},
        "port": 5,
    }
    text = render_annotated_config(config, toy_parser, "toy")
    assert yaml.safe_load(text) == config


def test_used_keys_come_first_and_the_rest_is_commented_out(toy_parser):
    text = render_annotated_config({"kind": "documented"}, toy_parser, "toy")
    top, bottom = _split(text)
    assert "\nkind: documented  # default: bare\n" in top
    assert "\n# width: 8\n" in bottom
    assert "width" not in top
    assert all(not line or line.startswith("#") for line in bottom.splitlines())


def test_hash_excluded_flags_stay_out_of_the_reference_section(toy_parser):
    _, bottom = _split(render_annotated_config({}, toy_parser, "toy"))
    assert "port" not in bottom


def test_used_key_carries_help_doc_and_value_description(toy_parser):
    top, _ = _split(
        render_annotated_config(
            {"kind": "documented", "kinds": ["profile"]}, toy_parser, "toy"
        )
    )
    assert "# Which kind to use.\n" in top
    assert "# The toy namespace's doc, the long form of every flag bound to it." in top
    assert "#   documented\n#     A documented entry.\n" in top
    assert "#   profile\n#     A profile with its own doc.\n" in top
    assert "Second paragraph" not in top
    assert "\nkinds: [profile]  # default: []\n" in top


def test_default_note_only_when_the_value_differs(toy_parser):
    top, _ = _split(render_annotated_config({"width": 8}, toy_parser, "toy"))
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
