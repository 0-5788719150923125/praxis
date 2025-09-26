"""Praxis CLI system - modular command-line interface."""

import sys

from .core import (
    DEFAULT_EXCLUDE_FROM_HASH,
    CustomHelpFormatter,
    compute_args_hash,
    create_base_parser,
    log_command,
)
from .groups import OtherGroup, add_all_argument_groups, process_all_arguments
from .loaders import EnvironmentLoader, ExperimentLoader, IntegrationBridge
from .processors import ArgumentProcessor, ConfigBuilder

# Global state (for backward compatibility)
parser = None
args = None
integration_loader = None
experiment_configs = {}
environment_configs = {}


def initialize_cli():
    """
    Initialize the CLI system and parse arguments.

    Returns:
        tuple: (parser, args, integration_loader)
    """
    global parser, args, integration_loader, experiment_configs, environment_configs

    # Create base parser
    parser = create_base_parser()

    # Load experiments
    experiment_loader = ExperimentLoader()
    experiment_configs = experiment_loader.load_experiments(parser)

    # Load environments
    environment_loader = EnvironmentLoader()
    environment_configs = environment_loader.load_environments(parser)

    # Initialize integration bridge
    integration_bridge = IntegrationBridge()
    integration_bridge.discover_and_bootstrap(experiment_configs, environment_configs)

    # Add all argument groups FIRST (so integrations can add to them)
    add_all_argument_groups(parser)

    # Add integration CLI arguments (they will add to existing groups)
    integration_bridge.add_cli_arguments(parser)

    # Add --dev if not already present (backward compatibility)
    OtherGroup.add_dev_argument_if_needed(parser)

    # Parse arguments
    args = parser.parse_args()

    # Get explicitly provided arguments
    explicitly_provided = ArgumentProcessor.get_explicitly_provided()

    # Apply experiments
    args = experiment_loader.apply_experiments(args, explicitly_provided)

    # Apply environments (highest priority)
    args, active_env, env_features = environment_loader.apply_environments(args)

    # Process arguments through all groups
    args = process_all_arguments(args)

    # Finalize integration loading
    integration_bridge.finalize_loading(args)

    # Get the integration loader for export
    integration_loader = integration_bridge.get_loader()

    return parser, args, integration_loader


def apply_defaults_and_parse(defaults_dict):
    """
    Apply custom default values to the parser and re-parse arguments.
    Used by scripts like run_alpha.py.

    Args:
        defaults_dict: Dictionary mapping argument names to default values

    Returns:
        tuple: (parsed_args, hash_computed_from_effective_config)
    """
    global parser, args, integration_loader

    # Ensure CLI is initialized
    if parser is None:
        initialize_cli()

    # Use processor to apply defaults
    args, effective_hash = ArgumentProcessor.apply_defaults_for_scripts(
        parser, defaults_dict, experiment_configs, environment_configs
    )

    # Re-finalize integrations with new args
    integration_bridge = IntegrationBridge()
    integration_bridge.loader = integration_loader
    integration_bridge.finalize_loading(args)

    return args, effective_hash


def create_praxis_config(args_override=None, tokenizer=None):
    """
    Create a PraxisConfig object from CLI arguments.

    Args:
        args_override: Optional args object (defaults to global args)
        tokenizer: Optional tokenizer for token IDs

    Returns:
        PraxisConfig: Configuration object ready for model initialization
    """
    if args_override is None:
        args_override = args

    return ConfigBuilder.create_praxis_config(args_override, tokenizer)


def get_processed_args(args_override=None):
    """
    Get a dictionary of processed arguments suitable for use in main.py.

    Args:
        args_override: Optional args object (defaults to global args)

    Returns:
        dict: Processed arguments ready for use
    """
    if args_override is None:
        args_override = args

    return ArgumentProcessor.process_arguments(args_override)


def get_cli_args():
    """Get the parsed CLI arguments."""
    return args


# Initialize CLI on module import (for backward compatibility)
if __name__ != "__main__":
    # Only auto-initialize if imported as a module
    if len(sys.argv) > 1:  # Don't initialize if no args provided
        parser, args, integration_loader = initialize_cli()

# Export public API
__all__ = [
    # Core functionality
    "initialize_cli",
    "apply_defaults_and_parse",
    "create_praxis_config",
    "get_processed_args",
    "get_cli_args",
    "log_command",
    "compute_args_hash",
    # Global state (for backward compatibility)
    "parser",
    "args",
    "integration_loader",
    # Constants
    "DEFAULT_EXCLUDE_FROM_HASH",
]
