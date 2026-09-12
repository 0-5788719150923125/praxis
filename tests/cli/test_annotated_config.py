"""Tests for praxis/cli/annotated_config.py, the documented config format behind
the web app's Download button and the rewrite of the committed experiments."""

import argparse
import subprocess

import pytest
import yaml

from praxis.cli.annotated_config import (
    _plain,
    format_tracked_experiments,
    reference_parser,
    render_annotated_config,
    render_experiment_file,
    tracked_experiments,
    undocumentable_keys,
)


def _split(text):
    return text.split("EVERYTHING ELSE, AT ITS DEFAULT", 1)


# ---------------------------------------------------------------------------
# the format
# ---------------------------------------------------------------------------


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
    assert "\nkind: documented\n" in top
    assert "\n# width: 8\n" in bottom
    assert "width" not in top
    assert all(not line or line.startswith("#") for line in bottom.splitlines())


def test_every_key_is_preceded_by_a_blank_line(toy_parser):
    text = render_annotated_config(
        {"kind": "documented", "width": 3}, toy_parser, "toy"
    )
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line and not line.startswith("#") and not line.startswith(" "):
            assert lines[i - 1] == "", f"{line!r} has no blank line above it"


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
    assert "# documented: A documented entry.\n" in top
    assert "# profile: A profile with its own doc.\n" in top
    assert "Second paragraph" not in top
    assert "\nkinds: [profile]\n" in top


def test_options_and_default_are_one_paragraph(toy_parser):
    top, _ = _split(render_annotated_config({"kind": "documented"}, toy_parser, "toy"))
    assert "# Options: documented, bare, profile\n# Default: bare\n" in top


def test_reference_entry_shows_the_default_on_its_key_line_only(toy_parser):
    _, bottom = _split(render_annotated_config({}, toy_parser, "toy"))
    entry = bottom.split("# How wide")[1].split("# The one sentence")[0]
    assert "# Default:" not in entry
    assert "\n# width: 8\n" in entry


def test_a_help_string_repeating_the_doc_is_not_printed_twice(toy_parser):
    top, _ = _split(render_annotated_config({"echo": 1}, toy_parser, "toy"))
    assert top.count("The one sentence both forms open with.") == 1
    assert "Only the long form says this." in top


def test_command_line_mechanics_are_dropped_from_the_prose(toy_parser):
    top, _ = _split(render_annotated_config({"listy": ["a"]}, toy_parser, "toy"))
    assert "# Things to list.\n" in top
    assert "space-separated" not in top
    assert "# Options: a, b, c\n" in top


def test_a_volatile_default_renders_as_a_placeholder(toy_parser):
    first = render_annotated_config({}, toy_parser, "toy")
    second = render_annotated_config({}, toy_parser, "toy")
    assert "# rolled: <int>" in first
    assert "# Default: drawn fresh each run" in first
    assert first == second


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


# ---------------------------------------------------------------------------
# rewriting a file
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    """A git repository with one tracked experiment and one gitignored one."""
    (tmp_path / "tracked.yml").write_text("# Why this arm exists.\nkind: documented\n")
    (tmp_path / "mine.yml").write_text("kind: bare\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tracked.yml"], cwd=tmp_path, check=True)
    return tmp_path


def test_only_tracked_files_are_listed(repo):
    assert [p.name for p in tracked_experiments(repo)] == ["tracked.yml"]


def test_a_directory_outside_a_repository_lists_nothing(tmp_path):
    (tmp_path / "loose.yml").write_text("kind: bare\n")
    assert tracked_experiments(tmp_path / "missing") == []


def test_gitignored_files_are_never_rewritten(repo, toy_parser):
    before = (repo / "mine.yml").read_text()
    format_tracked_experiments(toy_parser, repo)
    assert (repo / "mine.yml").read_text() == before


def test_the_rewrite_keeps_the_values_and_the_hand_written_note(repo, toy_parser):
    format_tracked_experiments(toy_parser, repo)
    text = (repo / "tracked.yml").read_text()
    assert yaml.safe_load(text)["kind"] == "documented"
    assert "# Note: Why this arm exists." in text


def test_the_rewrite_is_idempotent(repo, toy_parser):
    format_tracked_experiments(toy_parser, repo)
    once = (repo / "tracked.yml").read_text()
    assert format_tracked_experiments(toy_parser, repo) == []
    assert (repo / "tracked.yml").read_text() == once


def test_a_note_keeps_its_bullets_across_two_passes(repo, toy_parser):
    (repo / "tracked.yml").write_text(
        "# Traps:\n"
        "# * the first one - which wraps far enough to need a second line of "
        "prose here\n"
        "# * the second one\n"
        "kind: documented\n"
    )
    format_tracked_experiments(toy_parser, repo)
    once = (repo / "tracked.yml").read_text()
    assert once.count("* the first one") == 1
    assert once.count("* the second one") == 1
    assert format_tracked_experiments(toy_parser, repo) == []


def test_extends_survives_the_rewrite_unresolved(repo, toy_parser):
    (repo / "tracked.yml").write_text("extends: base\nkind: documented\n")
    format_tracked_experiments(toy_parser, repo)
    text = (repo / "tracked.yml").read_text()
    assert yaml.safe_load(text)["extends"] == "base"
    assert "\nextends: base\n" in text


def test_a_key_nothing_describes_blocks_the_rewrite(repo, toy_parser):
    (repo / "tracked.yml").write_text("kind: documented\nnobody_knows: 1\n")
    before = (repo / "tracked.yml").read_text()
    assert undocumentable_keys({"nobody_knows": 1}, toy_parser) == ["nobody_knows"]
    assert render_experiment_file(repo / "tracked.yml", toy_parser) is None
    assert format_tracked_experiments(toy_parser, repo) == []
    assert (repo / "tracked.yml").read_text() == before
