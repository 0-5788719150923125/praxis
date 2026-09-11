"""Tests for praxis/cli/loaders: PRAXIS_* environment overrides and the
``extends`` keyword in experiment YAML."""

import argparse
import os
import textwrap

import pytest

from praxis.cli.loaders.env_vars import EnvVarLoader
from praxis.cli.loaders.experiments import load_rendered_config

# ------------------------------------------------------------------------------
# environment variables
# ------------------------------------------------------------------------------


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


def _apply(monkeypatch, env, argv=(), explicit=()):
    parser = _build_parser()
    args = parser.parse_args(list(argv))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    loader = EnvVarLoader()
    loader.apply_env_vars(parser, args, explicitly_provided=set(explicit))
    return args, loader


@pytest.mark.parametrize(
    "env,attr,expected",
    [
        ({"PRAXIS_BATCH_SIZE": "32"}, "batch_size", 32),
        ({"PRAXIS_DEVICE": "cuda:1"}, "device", "cuda:1"),
        ({"PRAXIS_OPTIMIZER": "AdamW"}, "optimizer", "AdamW"),
        ({"PRAXIS_DATA_PATH": "/data/a, /data/b"}, "data_path", ["/data/a", "/data/b"]),
        (
            {"PRAXIS_DATA_PATH": '["/data/a", "/data/b"]'},
            "data_path",
            ["/data/a", "/data/b"],
        ),
        ({"PRAXIS_META": "one,two,three"}, "meta", ["one", "two", "three"]),
    ],
    ids=["int", "string", "choice", "nargs-comma", "nargs-json", "append"],
)
def test_env_var_sets_the_typed_value(monkeypatch, env, attr, expected):
    args, _ = _apply(monkeypatch, env)
    assert getattr(args, attr) == expected
    assert type(getattr(args, attr)) is type(expected)


@pytest.mark.parametrize(
    "raw,expected",
    [("1", True), ("true", True), ("YES", True), ("0", False), ("false", False)],
)
def test_env_var_store_true_bool(monkeypatch, raw, expected):
    args, _ = _apply(monkeypatch, {"PRAXIS_DEBUG": raw})
    assert args.debug is expected


@pytest.mark.parametrize(
    "env,attr,default,message",
    [
        (
            {"PRAXIS_OPTIMIZER": "NotARealOptimizer"},
            "optimizer",
            "Lion",
            "not in choices",
        ),
        ({"PRAXIS_BATCH_SIZE": "not-an-int"}, "batch_size", 1, ""),
    ],
    ids=["choice", "int"],
)
def test_an_invalid_value_keeps_the_default_and_names_the_var(
    monkeypatch, capsys, env, attr, default, message
):
    args, _ = _apply(monkeypatch, env)
    assert getattr(args, attr) == default
    out = capsys.readouterr().out
    assert next(iter(env)) in out
    assert message in out


@pytest.mark.parametrize(
    "env,argv,explicit,attr,expected",
    [
        (
            {"PRAXIS_BATCH_SIZE": "32"},
            ["--batch-size", "64"],
            {"batch-size"},
            "batch_size",
            64,
        ),
        # The explicit set may record either spelling; both are honoured.
        ({"PRAXIS_BATCH_SIZE": "32"}, [], {"batch_size"}, "batch_size", 1),
        ({"PRAXIS_DEBUG": "0"}, ["--debug"], {"debug"}, "debug", True),
    ],
    ids=["dashed", "underscored", "store-true"],
)
def test_an_explicit_cli_flag_wins_over_the_env(
    monkeypatch, env, argv, explicit, attr, expected
):
    args, _ = _apply(monkeypatch, env, argv, explicit)
    assert getattr(args, attr) == expected


def test_no_env_vars_is_noop(monkeypatch):
    for key in list(os.environ):
        if key.startswith("PRAXIS_"):
            monkeypatch.delenv(key)
    _, loader = _apply(monkeypatch, {})
    assert loader.applied == {}


# ------------------------------------------------------------------------------
# experiment extends
# ------------------------------------------------------------------------------


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

    override = _write(tmp_path, "override", "extends: [one, two]\nshared: child\n")
    assert load_rendered_config(override)["shared"] == "child"


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
