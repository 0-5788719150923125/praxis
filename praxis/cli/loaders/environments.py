"""Environment loading and management."""

import re
from pathlib import Path

import yaml


class EnvironmentLoader:
    """Loads and manages environment configurations."""

    def __init__(self, environments_dir="environments"):
        self.environments_dir = Path(environments_dir)
        self.configs = {}
        self.active_environment = None

    def load_environments(self, parser):
        """
        Load environment files from the environments directory and add them as CLI arguments.
        Each YAML file in environments/ becomes a --<filename> flag that applies those settings
        with highest priority (overwrites everything else).

        Returns:
            dict: Loaded environment configurations
        """
        if not self.environments_dir.exists():
            return {}

        # Get environment files
        environment_files = list(self.environments_dir.glob("*.yml"))
        if not environment_files:
            return {}

        # Create environments argument group
        environments_group = parser.add_argument_group("environments")

        for environment_file in environment_files:
            # Validate and normalize the environment name
            name = environment_file.stem.lower()

            # Only allow alphanumeric characters and hyphens
            if not re.match(r"^[a-z0-9-]+$", name):
                print(
                    f"Warning: Skipping environment '{environment_file.name}' - "
                    f"name must only contain lowercase letters, numbers, and hyphens"
                )
                continue

            # Check for conflicts with existing arguments
            arg_name = f"--{name}"
            if any(arg_name in action.option_strings for action in parser._actions):
                # Skip if argument already exists (like --dev from the 'other' group)
                # The existing argument will still work to activate the environment
                continue

            # Add new environment argument
            environments_group.add_argument(
                arg_name,
                action="store_true",
                default=False,
                help=f"Apply {name} environment configuration",
            )

            # Load and store the environment config
            try:
                with open(environment_file, "r") as f:
                    config = yaml.safe_load(f)
                    self.configs[name] = {
                        "overrides": config.get("overrides", {}),
                        "features": config.get("features", {}),
                    }
            except Exception as e:
                print(f"Warning: Failed to load environment '{environment_file.name}': {e}")
                continue

        return self.configs

    def apply_environments(self, args):
        """
        Apply environment configurations to parsed arguments.
        Environments have highest priority and override everything.

        Args:
            args: Parsed arguments object

        Returns:
            tuple: (modified args, active environment name, environment features)
        """
        from praxis.environments import EnvironmentFeatures

        active_env = None
        env_features = {}

        # Check for environment flags and apply them
        for env_name in self.configs:
            if getattr(args, env_name.replace("-", "_"), False):
                if active_env and active_env != env_name:
                    raise ValueError(
                        f"Cannot use multiple environments simultaneously: "
                        f"--{active_env} and --{env_name}"
                    )
                active_env = env_name

        # Apply the active environment's configuration if one is set
        if active_env and active_env in self.configs:
            env_config = self.configs[active_env]

            # Apply overrides (these have highest priority, overwrite everything)
            for key, value in env_config["overrides"].items():
                attr_name = key.replace("-", "_")
                setattr(args, attr_name, value)

            # Store features
            env_features = env_config["features"]

        # Set environment features globally
        EnvironmentFeatures.set_from_environment(env_features, active_env)

        self.active_environment = active_env
        return args, active_env, env_features

    def get_preliminary_configs(self, preliminary_args):
        """
        Apply environment configurations for preliminary parsing.

        Args:
            preliminary_args: Preliminary parsed arguments

        Returns:
            Active environment name if any
        """
        active_env = None

        for env_name in self.configs:
            if getattr(preliminary_args, env_name.replace("-", "_"), False):
                if active_env and active_env != env_name:
                    raise ValueError(
                        f"Cannot use multiple environments: --{active_env} and --{env_name}"
                    )
                active_env = env_name

                # Apply overrides
                for key, value in self.configs[env_name]["overrides"].items():
                    attr_name = key.replace("-", "_")
                    setattr(preliminary_args, attr_name, value)

        return active_env