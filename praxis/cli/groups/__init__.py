"""Argument groups for CLI organization."""

from .architecture import ArchitectureGroup
from .data import DataGroup
from .hardware import HardwareGroup
from .logging import LoggingGroup
from .networking import NetworkingGroup
from .optimization import OptimizationGroup
from .other import OtherGroup
from .persistence import PersistenceGroup
from .tokenizer import TokenizerGroup
from .training import TrainingGroup

# Registry of all argument groups
ARGUMENT_GROUPS = [
    HardwareGroup,
    PersistenceGroup,
    ArchitectureGroup,
    TrainingGroup,
    OptimizationGroup,
    NetworkingGroup,
    DataGroup,
    LoggingGroup,
    TokenizerGroup,
    OtherGroup,
]


def add_all_argument_groups(parser):
    """Add all registered argument groups to the parser."""
    for group_class in ARGUMENT_GROUPS:
        group_class.add_arguments(parser)


def process_all_arguments(args):
    """Process arguments through all groups that have processors."""
    for group_class in ARGUMENT_GROUPS:
        if hasattr(group_class, "process_args"):
            group_class.process_args(args)
    return args


__all__ = [
    "ARGUMENT_GROUPS",
    "add_all_argument_groups",
    "process_all_arguments",
    "HardwareGroup",
    "PersistenceGroup",
    "ArchitectureGroup",
    "TrainingGroup",
    "OptimizationGroup",
    "NetworkingGroup",
    "DataGroup",
    "LoggingGroup",
    "TokenizerGroup",
    "OtherGroup",
]