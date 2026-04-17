"""Tests for the `extends` keyword in experiment YAML loading."""

import importlib.util
import textwrap
from pathlib import Path

import pytest

# Load the experiments module directly so we don't trigger praxis.cli.__init__,
# which calls argparse.parse_args() at import time and chokes on pytest's argv.
_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "praxis"
    / "cli"
    / "loaders"
    / "experiments.py"
)
_spec = importlib.util.spec_from_file_location("_experiments_loader", _MODULE_PATH)
_experiments = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_experiments)
load_rendered_config = _experiments.load_rendered_config


def _write(dir_path, name, contents):
    path = dir_path / f"{name}.yml"
    path.write_text(textwrap.dedent(contents).lstrip())
    return path


def test_config_without_extends_is_unchanged(tmp_path):
    path = _write(tmp_path, "plain", "batch_size: 8\nseed: 1\n")
    assert load_rendered_config(path) == {"batch_size": 8, "seed": 1}


def test_single_extends_merges_and_strips_keyword(tmp_path):
    _write(tmp_path, "base", "batch_size: 8\nseed: 1\n")
    child = _write(tmp_path, "child", "extends: base\nseed: 99\n")

    rendered = load_rendered_config(child)

    assert rendered == {"batch_size": 8, "seed": 99}
    assert "extends" not in rendered


def test_extends_accepts_filename_with_yml_suffix(tmp_path):
    _write(tmp_path, "base", "a: 1\n")
    child = _write(tmp_path, "child", "extends: base.yml\nb: 2\n")
    assert load_rendered_config(child) == {"a": 1, "b": 2}


def test_chained_extends_resolves_full_chain(tmp_path):
    _write(tmp_path, "a", "x: 1\ny: 1\nz: 1\n")
    _write(tmp_path, "b", "extends: a\ny: 2\nz: 2\n")
    c = _write(tmp_path, "c", "extends: b\nz: 3\n")

    assert load_rendered_config(c) == {"x": 1, "y": 2, "z": 3}


def test_list_extends_merges_left_to_right(tmp_path):
    _write(tmp_path, "one", "a: 1\nshared: one\n")
    _write(tmp_path, "two", "b: 2\nshared: two\n")
    child = _write(
        tmp_path,
        "child",
        """
        extends:
          - one
          - two
        c: 3
        """,
    )

    # `two` beats `one`; child has no `shared`, so `two` wins.
    assert load_rendered_config(child) == {"a": 1, "b": 2, "shared": "two", "c": 3}


def test_child_overrides_all_bases(tmp_path):
    _write(tmp_path, "one", "shared: one\n")
    _write(tmp_path, "two", "shared: two\n")
    child = _write(
        tmp_path,
        "child",
        """
        extends: [one, two]
        shared: child
        """,
    )
    assert load_rendered_config(child)["shared"] == "child"


def test_nested_dicts_are_deep_merged(tmp_path):
    _write(
        tmp_path,
        "base",
        """
        nested:
          keep: kept
          override: parent
        """,
    )
    child = _write(
        tmp_path,
        "child",
        """
        extends: base
        nested:
          override: child
          added: new
        """,
    )
    assert load_rendered_config(child) == {
        "nested": {"keep": "kept", "override": "child", "added": "new"}
    }


def test_cycle_is_detected(tmp_path):
    _write(tmp_path, "a", "extends: b\n")
    _write(tmp_path, "b", "extends: a\n")

    with pytest.raises(ValueError, match="Circular 'extends'"):
        load_rendered_config(tmp_path / "a.yml")


def test_missing_base_raises(tmp_path):
    child = _write(tmp_path, "child", "extends: ghost\n")
    with pytest.raises(FileNotFoundError, match="ghost"):
        load_rendered_config(child)


def test_invalid_extends_type_raises(tmp_path):
    child = _write(tmp_path, "child", "extends: 42\n")
    with pytest.raises(ValueError, match="'extends' must be"):
        load_rendered_config(child)


def test_repository_alpha_experiment_loads():
    # Sanity check: the real alpha.yml (no extends) still loads cleanly.
    alpha = Path(__file__).resolve().parents[1] / "experiments" / "alpha.yml"
    rendered = load_rendered_config(alpha)
    assert "extends" not in rendered
    assert rendered.get("batch_size") == 16
