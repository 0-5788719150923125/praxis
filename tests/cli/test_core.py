"""Tests for praxis/cli/core: registry-bound flags on the parser, and the args hash."""

import pytest

from praxis.cli.core import create_base_parser
from praxis.cli.core.hasher import compute_args_hash


def test_registry_sets_the_choices(toy_parser):
    assert toy_parser.parse_args(["--kind", "profile"]).kind == "profile"
    with pytest.raises(SystemExit):
        toy_parser.parse_args(["--kind", "nope"])


def test_a_registry_flag_cannot_carry_its_own_doc(toy_registry):
    group = create_base_parser().add_argument_group("toy")
    with pytest.raises(ValueError, match="document it where the namespace"):
        group.add_argument("--kind", registry=toy_registry, doc="a second copy")


def test_doc_stays_out_of_help(toy_parser):
    assert "long form" not in toy_parser.format_help()


def test_generation_mode_is_not_part_of_the_model_hash():
    """Both decoding paths are trained by the same objectives, so
    ``--generation-mode`` is inference-only and must not fork the model hash
    (a registry profile would have forced a separate training run)."""
    base = ["--encoder-type", "x", "--batch-size", "4"]
    assert compute_args_hash(base) == compute_args_hash(
        base + ["--generation-mode", "vote"]
    )
