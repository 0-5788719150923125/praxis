"""Experiment loading and management."""

import re
from pathlib import Path
from typing import Optional

import yaml

EXTENDS_KEY = "extends"


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base. Override wins; nested dicts recurse."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_rendered_config(
    config_path,
    experiments_dir=None,
    _visited: Optional[set] = None,
) -> dict:
    """
    Load an experiment YAML file and recursively resolve its `extends` keyword.

    `extends` may be a single experiment name or a list of names. Names refer
    to sibling YAML files by stem (e.g. `extends: alpha` -> `alpha.yml`).
    When a list is given, entries are merged left-to-right; the current file
    then overrides all bases. The returned dict never contains `extends`.
    """
    config_path = Path(config_path)
    if experiments_dir is None:
        experiments_dir = config_path.parent
    experiments_dir = Path(experiments_dir)

    resolved_path = config_path.resolve()
    if _visited is None:
        _visited = set()
    if resolved_path in _visited:
        chain = " -> ".join(str(p) for p in sorted(_visited)) + f" -> {resolved_path}"
        raise ValueError(f"Circular 'extends' detected: {chain}")
    next_visited = _visited | {resolved_path}

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Experiment config must be a mapping: {config_path}")

    extends = raw.pop(EXTENDS_KEY, None)
    if extends is None:
        return raw

    if isinstance(extends, str):
        bases = [extends]
    elif isinstance(extends, list):
        bases = extends
    else:
        raise ValueError(
            f"'extends' must be a string or list of strings in {config_path}"
        )

    merged: dict = {}
    for base in bases:
        if not isinstance(base, str):
            raise ValueError(f"'extends' entries must be strings in {config_path}")
        # Accept either a stem ("november") or a filename ("november.yml").
        stem = base[:-4] if base.endswith(".yml") else base
        base_path = experiments_dir / f"{stem}.yml"
        if not base_path.exists():
            raise FileNotFoundError(
                f"'extends' target '{base}' not found at {base_path} "
                f"(referenced from {config_path})"
            )
        base_config = load_rendered_config(
            base_path, experiments_dir=experiments_dir, _visited=next_visited
        )
        merged = _deep_merge(merged, base_config)

    return _deep_merge(merged, raw)


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

            # Load and store the experiment config (extends resolved, keyword stripped)
            try:
                self.configs[name] = load_rendered_config(
                    experiment_file, experiments_dir=self.experiments_dir
                )
            except Exception as e:
                print(
                    f"Warning: Failed to load experiment '{experiment_file.name}': {e}"
                )
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
                # Store the experiment config file path
                config_file_path = self.experiments_dir / f"{experiment_name}.yml"
                args.config_file = str(config_file_path)

                # Apply experiment defaults (but don't override user-provided values)
                for key, value in config.items():
                    attr_name = key.replace("-", "_")

                    # Check if this argument was explicitly provided by the user
                    if (
                        key in explicitly_provided
                        or key.replace("_", "-") in explicitly_provided
                    ):
                        continue  # User override takes precedence

                    # Apply the experiment default
                    setattr(args, attr_name, value)

        return args
