"""Argument processing and transformation."""

import os
import sys


class ArgumentProcessor:
    """Processes and transforms parsed arguments."""

    @staticmethod
    def get_explicitly_provided(argv=None):
        """
        Determine which arguments were explicitly provided by the user.

        Args:
            argv: Command line arguments (defaults to sys.argv)

        Returns:
            set: Set of explicitly provided argument names
        """
        if argv is None:
            argv = sys.argv[1:]  # Skip script name

        explicitly_provided = set()
        for arg in argv:
            if arg.startswith("--") and "=" in arg:
                # Handle --arg=value format
                key = arg.split("=")[0][2:]
                explicitly_provided.add(key)
            elif arg.startswith("--"):
                # Handle --arg value format
                key = arg[2:]
                explicitly_provided.add(key)

        return explicitly_provided

    @staticmethod
    def process_arguments(args):
        """
        Process parsed arguments with computed values and transformations.

        Args:
            args: Parsed arguments object

        Returns:
            dict: Processed arguments ready for use
        """
        # Create a dict from args
        processed = vars(args).copy()

        # Add computed values
        processed["byte_latent"] = (
            args.encoder_type == "byte_latent" if hasattr(args, "encoder_type") else False
        )

        # Adjust block_size for byte_latent
        if processed["byte_latent"] and "block_size" in processed:
            processed["block_size"] = processed["block_size"] * 8

        # Set terminal_output_length
        if "block_size" in processed:
            if processed["byte_latent"]:
                processed["terminal_output_length"] = processed["block_size"] // 2
            else:
                processed["terminal_output_length"] = processed["block_size"] * 2

        # Set use_dashboard
        processed["use_dashboard"] = not processed.get("no_dashboard", False)

        # Set local_rank
        processed["local_rank"] = int(os.environ.get("LOCAL_RANK", 0))

        return processed

    @staticmethod
    def apply_defaults_for_scripts(parser, defaults_dict, experiment_configs, environment_configs):
        """
        Apply custom defaults for scripts like run_alpha.py.

        Args:
            parser: The argument parser
            defaults_dict: Dictionary of default values to apply
            experiment_configs: Loaded experiment configurations
            environment_configs: Loaded environment configurations

        Returns:
            tuple: (args, computed_hash)
        """
        from praxis.cli.core import compute_args_hash, log_command, DEFAULT_EXCLUDE_FROM_HASH
        from praxis.environments import EnvironmentFeatures

        # Store original command for logging
        original_command = sys.argv[:]

        # Apply custom defaults to parser
        for action in parser._actions:
            if hasattr(action, "dest") and action.dest in defaults_dict:
                action.default = defaults_dict[action.dest]

        # Re-parse arguments with new defaults
        args = parser.parse_args()

        # Apply experiment defaults if any experiment flags are set
        if experiment_configs:
            for experiment_name, config in experiment_configs.items():
                if getattr(args, experiment_name.replace("-", "_"), False):
                    for key, value in config.items():
                        attr_name = key.replace("-", "_")
                        # Check if this argument was explicitly provided by the user
                        if attr_name in sys.argv or f"--{key}" in sys.argv:
                            continue
                        setattr(args, attr_name, value)

        # Apply environment overrides
        active_env = None
        env_features = {}

        # Check for environment activation
        for env_name in environment_configs:
            if getattr(args, env_name.replace("-", "_"), False):
                if active_env:
                    raise ValueError(
                        f"Cannot use multiple environments: --{active_env} and --{env_name}"
                    )
                active_env = env_name
                env_features = environment_configs[env_name]["features"]
                for key, value in environment_configs[env_name]["overrides"].items():
                    setattr(args, key.replace("-", "_"), value)

        EnvironmentFeatures.set_from_environment(env_features, active_env)

        # Build equivalent command for hash computation
        equivalent_args = []

        # Convert defaults to CLI format for hash computation
        for arg_name, value in defaults_dict.items():
            cli_arg = "--" + arg_name.replace("_", "-")
            if isinstance(value, bool):
                if value:  # Only add flag if it's True
                    equivalent_args.append(cli_arg)
            else:
                equivalent_args.extend([cli_arg, str(value)])

        # Get user-provided arguments
        user_args = sys.argv[1:]
        user_arg_names = set()
        i = 0
        while i < len(user_args):
            if user_args[i].startswith("--"):
                user_arg_names.add(user_args[i])
                if i + 1 < len(user_args) and not user_args[i + 1].startswith("-"):
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        # Filter out defaults that user has overridden
        filtered_defaults = []
        i = 0
        while i < len(equivalent_args):
            if equivalent_args[i] not in user_arg_names:
                filtered_defaults.append(equivalent_args[i])
                if i + 1 < len(equivalent_args) and not equivalent_args[i + 1].startswith("-"):
                    filtered_defaults.append(equivalent_args[i + 1])
                    i += 2
                else:
                    i += 1
            else:
                if i + 1 < len(equivalent_args) and not equivalent_args[i + 1].startswith("-"):
                    i += 2
                else:
                    i += 1

        # Combine for hash computation
        hash_args = filtered_defaults + user_args

        # Compute hash from effective configuration
        effective_hash = compute_args_hash(hash_args)

        # Custom logging
        script_name = os.path.basename(original_command[0])
        args_list = original_command[1:]
        displayed_command = f"python {script_name} {' '.join(args_list)}"

        # Log with computed hash
        log_command(
            exclude_from_hash=DEFAULT_EXCLUDE_FROM_HASH,
            custom_command=displayed_command,
            custom_hash=effective_hash,
        )

        return args, effective_hash