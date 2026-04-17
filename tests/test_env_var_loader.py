"""Tests for PRAXIS_* environment variable overrides of CLI arguments."""

import argparse
import importlib.util
from pathlib import Path

import pytest

# Load the module directly so importing praxis.cli (which auto-parses sys.argv)
# doesn't fight with pytest's own argv.
_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "praxis" / "cli" / "loaders" / "env_vars.py"
)
_spec = importlib.util.spec_from_file_location("_env_var_loader", _MODULE_PATH)
_env_vars = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_env_vars)
EnvVarLoader = _env_vars.EnvVarLoader


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
