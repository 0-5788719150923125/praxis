"""Experiment loading and management."""

import re
from pathlib import Path

import yaml


class ExperimentLoader:
    """Loads and manages experiment configurations."""

    def __init__(self, experiments_dir="experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.configs = {}

    def load_experiments(self, parser):
        """
        Load experiment files from the experiments directory and add them as CLI arguments.
        Each YAML file in experiments/ becomes a --<filename> flag that applies those defaults.

        Returns:
            dict: Loaded experiment configurations
        """
        if not self.experiments_dir.exists():
            return {}

        # Get experiment files
        experiment_files = list(self.experiments_dir.glob("*.yml"))
        if not experiment_files:
            return {}

        # Create experiments argument group
        experiments_group = parser.add_argument_group("experiments")

        for experiment_file in experiment_files:
            # Validate and normalize the experiment name
            name = experiment_file.stem.lower()

            # Only allow alphanumeric characters and hyphens
            if not re.match(r"^[a-z0-9-]+$", name):
                print(
                    f"Warning: Skipping experiment '{experiment_file.name}' - "
                    f"name must only contain lowercase letters, numbers, and hyphens"
                )
                continue

            # Check for conflicts with existing arguments
            arg_name = f"--{name}"
            if any(arg_name in action.option_strings for action in parser._actions):
                print(
                    f"Warning: Skipping experiment '{experiment_file.name}' - "
                    f"conflicts with existing argument {arg_name}"
                )
                continue

            # Add the experiment argument
            experiments_group.add_argument(
                arg_name,
                action="store_true",
                default=False,
                help=f"Apply {name} experiment configuration",
            )

            # Load and store the experiment config
            try:
                with open(experiment_file, "r") as f:
                    self.configs[name] = yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Failed to load experiment '{experiment_file.name}': {e}")
                continue

        return self.configs

    def apply_experiments(self, args, explicitly_provided=None):
        """
        Apply experiment configurations to parsed arguments.

        Args:
            args: Parsed arguments object
            explicitly_provided: Set of explicitly provided argument names

        Returns:
            Modified args object
        """
        if explicitly_provided is None:
            explicitly_provided = set()

        for experiment_name, config in self.configs.items():
            if getattr(args, experiment_name.replace("-", "_"), False):
                # Apply experiment defaults (but don't override user-provided values)
                for key, value in config.items():
                    attr_name = key.replace("-", "_")

                    # Check if this argument was explicitly provided by the user
                    if key in explicitly_provided or key.replace("_", "-") in explicitly_provided:
                        continue  # User override takes precedence

                    # Apply the experiment default
                    setattr(args, attr_name, value)

        return args