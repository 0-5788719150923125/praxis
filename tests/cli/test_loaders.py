import argparse
import textwrap

import pytest

from praxis.cli.loaders.env_vars import EnvVarLoader
from praxis.cli.loaders.experiments import load_rendered_config

# ------------------------------------------------------------------------------
# env_var_loader
# ------------------------------------------------------------------------------
# Tests for PRAXIS_* environment variable overrides of CLI arguments.


def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--debug", action="store_true", default=False)
    p.add_argument("--no-dashboard", action="store_true", default=False)
    p.add_argument(
        "--optimizer", type=str, default="Lion", choices=["Lion", "AdamW", "SGD"]
    )
    p.add_argument("--data-path", type=str, nargs="+", action="extend", default=None)
    p.add_argument("--meta", action="append", default=[])
    return p


def _parse(parser, argv):
    return parser.parse_args(argv)


def test_env_var_sets_int(monkeypatch):
    parser = _build_parser()
    args = _parse(parser, [])
    monkeypatch.setenv("PRAXIS_BATCH_SIZE", "32")

    EnvVarLoader().apply_env_vars(parser, args, explicitly_provided=set())
    assert args.batch_size == 32
    assert isinstance(args.batch_size, int)


def test_env_var_sets_string(monkeypatch):
    parser = _build_parser()
    args = _parse(parser, [])
    monkeypatch.setenv("PRAXIS_DEVICE", "cuda:1")

    EnvVarLoader().apply_env_vars(parser, args, explicitly_provided=set())
    assert args.device == "cuda:1"


@pytest.mark.parametrize(
    "raw,expected",
    [("1", True), ("true", True), ("YES", True), ("0", False), ("false", False)],
)
def test_env_var_store_true_bool(monkeypatch, raw, expected):
    parser = _build_parser()
    args = _parse(parser, [])
    monkeypatch.setenv("PRAXIS_DEBUG", raw)

    EnvVarLoader().apply_env_vars(parser, args, explicitly_provided=set())
    assert args.debug is expected


def test_env_var_choices_valid(monkeypatch):
    parser = _build_parser()
    args = _parse(parser, [])
    monkeypatch.setenv("PRAXIS_OPTIMIZER", "AdamW")

    EnvVarLoader().apply_env_vars(parser, args, explicitly_provided=set())
    assert args.optimizer == "AdamW"


def test_env_var_choices_invalid_skipped(monkeypatch, capsys):
    parser = _build_parser()
    args = _parse(parser, [])
    monkeypatch.setenv("PRAXIS_OPTIMIZER", "NotARealOptimizer")

    EnvVarLoader().apply_env_vars(parser, args, explicitly_provided=set())
    # Unchanged — we refuse to apply an invalid value
    assert args.optimizer == "Lion"
    out = capsys.readouterr().out
    assert "PRAXIS_OPTIMIZER" in out
    assert "not in choices" in out


def test_env_var_nargs_list_comma(monkeypatch):
    parser = _build_parser()
    args = _parse(parser, [])
    monkeypatch.setenv("PRAXIS_DATA_PATH", "/data/a, /data/b")

    EnvVarLoader().apply_env_vars(parser, args, explicitly_provided=set())
    assert args.data_path == ["/data/a", "/data/b"]


def test_env_var_nargs_list_json(monkeypatch):
    parser = _build_parser()
    args = _parse(parser, [])
    monkeypatch.setenv("PRAXIS_DATA_PATH", '["/data/a", "/data/b"]')

    EnvVarLoader().apply_env_vars(parser, args, explicitly_provided=set())
    assert args.data_path == ["/data/a", "/data/b"]


def test_env_var_append_list(monkeypatch):
    parser = _build_parser()
    args = _parse(parser, [])
    monkeypatch.setenv("PRAXIS_META", "one,two,three")

    EnvVarLoader().apply_env_vars(parser, args, explicitly_provided=set())
    assert args.meta == ["one", "two", "three"]


def test_cli_explicit_wins_over_env(monkeypatch):
    parser = _build_parser()
    args = _parse(parser, ["--batch-size", "64"])
    monkeypatch.setenv("PRAXIS_BATCH_SIZE", "32")

    EnvVarLoader().apply_env_vars(parser, args, explicitly_provided={"batch-size"})
    assert args.batch_size == 64


def test_cli_explicit_underscore_form(monkeypatch):
    parser = _build_parser()
    args = _parse(parser, [])
    monkeypatch.setenv("PRAXIS_BATCH_SIZE", "32")

    # Explicit set may be recorded as either form; loader must honour both.
    EnvVarLoader().apply_env_vars(parser, args, explicitly_provided={"batch_size"})
    assert args.batch_size == 1


def test_invalid_int_skipped_with_warning(monkeypatch, capsys):
    parser = _build_parser()
    args = _parse(parser, [])
    monkeypatch.setenv("PRAXIS_BATCH_SIZE", "not-an-int")

    EnvVarLoader().apply_env_vars(parser, args, explicitly_provided=set())
    assert args.batch_size == 1  # unchanged
    assert "PRAXIS_BATCH_SIZE" in capsys.readouterr().out


def test_no_env_vars_is_noop(monkeypatch):
    parser = _build_parser()
    args = _parse(parser, [])
    # Make sure no PRAXIS_* leaked in from the shell.
    for key in list(__import__("os").environ):
        if key.startswith("PRAXIS_"):
            monkeypatch.delenv(key, raising=False)

    loader = EnvVarLoader()
    loader.apply_env_vars(parser, args, explicitly_provided=set())
    assert loader.applied == {}


def test_store_true_false_value(monkeypatch):
    parser = _build_parser()
    # Flip default True -> argparse default is False for store_true.
    args = _parse(parser, ["--debug"])  # CLI-explicit True
    monkeypatch.setenv("PRAXIS_DEBUG", "0")

    # CLI wins when explicit.
    EnvVarLoader().apply_env_vars(parser, args, explicitly_provided={"debug"})
    assert args.debug is True


# ------------------------------------------------------------------------------
# experiment_extends
# ------------------------------------------------------------------------------
# Tests for the `extends` keyword in experiment YAML loading.


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
