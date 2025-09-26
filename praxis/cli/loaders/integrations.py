"""Integration system bridge for CLI."""

import argparse

from praxis.integrations import IntegrationLoader


class IntegrationBridge:
    """Manages integration loading and CLI argument addition."""

    def __init__(self):
        self.loader = IntegrationLoader()
        self.integrations = []

    def discover_and_bootstrap(self, experiment_configs, environment_configs):
        """
        Discover integrations and perform preliminary bootstrap.

        Args:
            experiment_configs: Loaded experiment configurations
            environment_configs: Loaded environment configurations

        Returns:
            None
        """
        # Discover available integrations
        self.integrations = self.loader.discover_integrations()

        # Create preliminary parser for condition checking
        preliminary_parser = argparse.ArgumentParser(add_help=False)

        # Add experiment flags
        for experiment_name in experiment_configs:
            preliminary_parser.add_argument(
                f"--{experiment_name}", action="store_true", default=False
            )

        # Add environment flags
        for env_name in environment_configs:
            preliminary_parser.add_argument(
                f"--{env_name}", action="store_true", default=False
            )

        # Add known integration flags for condition checking
        for integration_manifest in self.integrations:
            integration_name = integration_manifest.name
            preliminary_parser.add_argument(
                f"--{integration_name}", action="store_true", default=False
            )

        # Parse known args (ignore unknown args from integrations)
        preliminary_args, _ = preliminary_parser.parse_known_args()

        # Apply experiments to preliminary args (medium priority)
        for experiment_name, config in experiment_configs.items():
            if getattr(preliminary_args, experiment_name.replace("-", "_"), False):
                for key, value in config.items():
                    attr_name = key.replace("-", "_")
                    setattr(preliminary_args, attr_name, value)

        # Apply environment overrides to preliminary args (highest priority)
        active_env = None
        for env_name in environment_configs:
            if getattr(preliminary_args, env_name.replace("-", "_"), False):
                if active_env and active_env != env_name:
                    raise ValueError(
                        f"Cannot use multiple environments: --{active_env} and --{env_name}"
                    )
                active_env = env_name
                # Apply overrides
                for key, value in environment_configs[env_name]["overrides"].items():
                    attr_name = key.replace("-", "_")
                    setattr(preliminary_args, attr_name, value)

        # Bootstrap only integrations whose conditions are met
        self.loader.bootstrap_integrations(preliminary_args)

        # Load all integrations (without condition checks yet)
        for integration_manifest in self.integrations:
            self.loader.load_integration(integration_manifest, verbose=False)

    def add_cli_arguments(self, parser):
        """Add integration CLI arguments to the parser."""
        for cli_func in self.loader.get_cli_functions():
            cli_func(parser)

    def finalize_loading(self, args):
        """
        Finalize integration loading with parsed arguments.

        Args:
            args: Final parsed arguments

        Returns:
            None
        """
        # Check integration conditions based on final args
        for integration_manifest in self.integrations:
            self.loader.load_integration(integration_manifest, args, verbose=False)

        # Print summary
        self.loader.print_summary()

    def get_loader(self):
        """Get the underlying integration loader."""
        return self.loader
