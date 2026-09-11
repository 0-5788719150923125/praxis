"""Tests for praxis/cli/groups: the argument groups' registry-bound flags."""

import pytest

from praxis.cli.core import create_base_parser
from praxis.cli.groups.architecture import ArchitectureGroup


def test_unlisted_encoder_name_is_accepted_on_the_command_line():
    parser = create_base_parser()
    ArchitectureGroup.add_arguments(parser)
    name = "abstractinator_harmonic_gdn_vocab_bank_static"
    assert parser.parse_args(["--encoder-type", name]).encoder_type == name
    with pytest.raises(SystemExit):
        parser.parse_args(["--encoder-type", "no_such_encoder"])
