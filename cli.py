import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

from praxis import (
    ACTIVATION_REGISTRY,
    ATTENTION_REGISTRY,
    BLOCK_REGISTRY,
    COMPRESSION_REGISTRY,
    CONTROLLER_REGISTRY,
    DECODER_REGISTRY,
    ENCODER_REGISTRY,
    ENCODING_REGISTRY,
    EXPERT_REGISTRY,
    HEAD_REGISTRY,
    LOSS_REGISTRY,
    NORMALIZATION_REGISTRY,
    RESIDUAL_REGISTRY,
    RL_POLICIES_REGISTRY,
    ROUTER_REGISTRY,
    SORTING_REGISTRY,
    STRATEGIES_REGISTRY,
)
from praxis.integrations import IntegrationLoader
from praxis.optimizers import OPTIMIZER_PROFILES
from praxis.trainers import TRAINER_REGISTRY

# Define the default list of arguments to exclude from hash computation
# These are typically runtime/debugging flags that don't affect model architecture
DEFAULT_EXCLUDE_FROM_HASH = [
    "--reset",
    "--debug",
    "--ngrok",
    "--wandb",
    "--no-dashboard",
]


def wrap_green(text):
    return f"\033[92m{text}\033[00m"


# User args, accepted via CLI
class CustomHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog, indent_increment=2, max_help_position=30, width=None):
        # Use terminal width if available, otherwise use a sensible default
        if width is None:
            try:
                import shutil

                width = shutil.get_terminal_size().columns
            except (ImportError, AttributeError):
                width = 100

        # Adjust max_help_position based on terminal width
        max_help_position = min(30, width // 3)

        super().__init__(prog, indent_increment, max_help_position, width)

    def _format_usage(self, usage, actions, groups, prefix):
        return ""  # This effectively removes the usage section

    def _format_action_invocation(self, action):
        """Customizes how arguments are displayed in the help output."""
        if action.option_strings:
            # It's an optional argument
            if action.nargs == 0:
                # It's a flag (like --verbose)
                return ", ".join(action.option_strings)
            else:
                # It takes a value (like --file <value>)
                return f"{', '.join(action.option_strings)} <value>"

    def _get_help_string(self, action):
        help_text = action.help or ""

        # Add type information when available
        if action.type is not None and hasattr(action.type, "__name__"):
            type_name = action.type.__name__
            if str(type_name) == "<lambda>":
                type_name = "str"
            help_text = f"({wrap_green(type_name)}) {help_text}"
        elif isinstance(action, argparse._StoreTrueAction) or isinstance(
            action, argparse._StoreFalseAction
        ):
            # It's a boolean flag
            help_text = f"({wrap_green('bool')}) {help_text}"

        # Add choices information when available (but only in the help text)
        if action.choices is not None:
            choice_str = ", ".join([str(c) for c in action.choices])
            help_text = f"{help_text} (choices: {choice_str})"

        # Add default value information when available
        if action.default is not argparse.SUPPRESS:
            # Always show default, even if it's None
            help_text = f"{help_text} (default: {str(action.default)})"

        return help_text


parser = argparse.ArgumentParser(
    description="Praxis CLI",
    formatter_class=CustomHelpFormatter,
)


hardware_group = parser.add_argument_group("hardware")
persistence_group = parser.add_argument_group("persistence")
architecture_group = parser.add_argument_group("architecture")
hparam_group = parser.add_argument_group("hyperparameters")
optimization_group = parser.add_argument_group("optimization")
networking_group = parser.add_argument_group("networking")
data_group = parser.add_argument_group("data")
other_group = parser.add_argument_group("other")


# Define functions to load experiments and environments
def load_experiments():
    """
    Load experiment files from the experiments directory and add them as CLI arguments.
    Each YAML file in experiments/ becomes a --<filename> flag that applies those defaults.
    """
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return {}

    # Create experiments argument group if we have any experiments
    experiment_files = list(experiments_dir.glob("*.yml"))
    if not experiment_files:
        return {}

    experiments_group = parser.add_argument_group("experiments")
    experiment_configs = {}

    for experiment_file in experiment_files:
        # Validate and normalize the experiment name
        name = experiment_file.stem.lower()

        # Only allow alphanumeric characters and hyphens
        if not re.match(r"^[a-z0-9-]+$", name):
            print(
                f"Warning: Skipping experiment '{experiment_file.name}' - name must only contain lowercase letters, numbers, and hyphens"
            )
            continue

        # Check for conflicts with existing arguments
        arg_name = f"--{name}"
        if any(arg_name in action.option_strings for action in parser._actions):
            print(
                f"Warning: Skipping experiment '{experiment_file.name}' - conflicts with existing argument {arg_name}"
            )
            continue

        # Add the experiment argument to the experiments group
        experiments_group.add_argument(
            arg_name,
            action="store_true",
            default=False,
            help=f"Apply {name} experiment configuration",
        )

        # Store the experiment config for later application
        try:
            with open(experiment_file, "r") as f:
                experiment_configs[name] = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Failed to load experiment '{experiment_file.name}': {e}")
            continue

    return experiment_configs


def load_environments():
    """
    Load environment files from the environments directory and add them as CLI arguments.
    Each YAML file in environments/ becomes a --<filename> flag that applies those settings
    with highest priority (overwrites everything else).
    """
    environments_dir = Path("environments")
    if not environments_dir.exists():
        return {}

    # Get environment files
    environment_files = list(environments_dir.glob("*.yml"))
    if not environment_files:
        return {}

    # Create environments argument group
    environments_group = parser.add_argument_group("environments")
    environment_configs = {}

    for environment_file in environment_files:
        # Validate and normalize the environment name
        name = environment_file.stem.lower()

        # Only allow alphanumeric characters and hyphens
        if not re.match(r"^[a-z0-9-]+$", name):
            print(
                f"Warning: Skipping environment '{environment_file.name}' - name must only contain lowercase letters, numbers, and hyphens"
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

        # Store the environment config for later application
        try:
            with open(environment_file, "r") as f:
                config = yaml.safe_load(f)
                environment_configs[name] = {
                    "overrides": config.get("overrides", {}),
                    "features": config.get("features", {}),
                }
        except Exception as e:
            print(f"Warning: Failed to load environment '{environment_file.name}': {e}")
            continue

    return environment_configs


# Load experiments and environments FIRST (before integration loading)
# We need these to properly determine which integrations should be loaded
experiment_configs = load_experiments()
environment_configs = load_environments()

# Initialize integration loader and discover modules
integration_loader = IntegrationLoader()
integrations = integration_loader.discover_integrations()

# Do a preliminary parse to check which integrations are needed
# We need to know this BEFORE installing dependencies
preliminary_parser = argparse.ArgumentParser(add_help=False)

# Add experiment flags to preliminary parser
for experiment_name in experiment_configs:
    preliminary_parser.add_argument(
        f"--{experiment_name}", action="store_true", default=False
    )

# Add environment flags to preliminary parser
for env_name in environment_configs:
    preliminary_parser.add_argument(f"--{env_name}", action="store_true", default=False)

# Add known integration flags for condition checking
for integration_manifest in integrations:
    # Add basic flag for each integration (e.g., --ngrok, --wandb)
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

# Bootstrap only integrations whose conditions are met (after applying all overrides)
integration_loader.bootstrap_integrations(preliminary_args)

# Now load integrations that can add CLI arguments (without condition checks yet)
for integration_manifest in integrations:
    integration_loader.load_integration(integration_manifest, verbose=False)

# Let integrations add CLI arguments
for cli_func in integration_loader.get_cli_functions():
    cli_func(parser)

# hardware
hardware_group.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to use",
)
hardware_group.add_argument(
    "--host-name",
    type=str,
    default="localhost",
    help="Serve the local API at this CNAME",
)
hardware_group.add_argument(
    "--port",
    type=int,
    default=2100,
    help="Serve the local API at this port",
)
hardware_group.add_argument(
    "--batch-size",
    type=int,
    default=1,
    help="Batch size to use for training",
)
hardware_group.add_argument(
    "--checkpoint-every",
    type=int,
    default=None,
    help="Apply gradient checkpointing every X layers",
)

# training
training_group = parser.add_argument_group("training")
training_group.add_argument(
    "--trainer-type",
    type=str,
    choices=list(TRAINER_REGISTRY.keys()),
    default="backpropagation",
    help="Training strategy to use (backpropagation: standard gradient descent, mono_forward: process-based pipeline parallelism with O(1) memory)",
)
# Mono-Forward specific arguments
training_group.add_argument(
    "--mono-forward-prediction-mode",
    type=str,
    choices=["ff", "bp"],
    default="bp",
    help="Mono-Forward prediction mode: 'ff' sums all layer goodness scores, 'bp' uses only last layer (default: bp)",
)
training_group.add_argument(
    "--mono-forward-vocab-reduction",
    type=int,
    default=4,
    help="Factor to reduce vocabulary size for internal Mono-Forward layers (default: 4)",
)
training_group.add_argument(
    "--max-steps",
    type=int,
    default=None,
    help="Maximum number of training steps (None for infinite training)",
)
training_group.add_argument(
    "--pipeline-depth",
    type=int,
    default=4,
    help="Number of batches to keep in pipeline for mono_forward trainer",
)

# storage
persistence_group.add_argument(
    "--cache-dir",
    type=str,
    default="build",
    help="Paths to a directory where artifacts will be saved",
)
# architecture
architecture_group.add_argument(
    "--encoder-type",
    type=str,
    choices=list(ENCODER_REGISTRY.keys()),
    default=None,
    help="Encoder integration to use",
)
architecture_group.add_argument(
    "--decoder-type",
    type=str,
    choices=list(DECODER_REGISTRY.keys()),
    default="sequential",
    help="How to process layers in the decoder",
)
architecture_group.add_argument(
    "--block-type",
    type=str,
    choices=BLOCK_REGISTRY.keys(),
    default="transformer",
    help="The type of block to use for every intermediate decoder layer",
)
architecture_group.add_argument(
    "--expert-type",
    type=str,
    choices=EXPERT_REGISTRY.keys(),
    default="glu",
    help="The integration to use for feedforward networks",
)
architecture_group.add_argument(
    "--attention-type",
    type=str,
    choices=ATTENTION_REGISTRY.keys(),
    default="standard",
    help="The base attention implementation to use",
)
architecture_group.add_argument(
    "--encoding-type",
    type=str,
    choices=ENCODING_REGISTRY.keys(),
    default="rope",
    help="The positional encoding to use for sequence length extrapolation",
)
architecture_group.add_argument(
    "--controller-type",
    type=str,
    choices=CONTROLLER_REGISTRY.keys(),
    default="base",
    help="Various methods used to route inputs through experts in the decoder",
)
architecture_group.add_argument(
    "--router-type",
    type=str,
    choices=ROUTER_REGISTRY.keys(),
    default=None,
    help="How to route tokens at every layer",
)
architecture_group.add_argument(
    "--residual-type",
    type=str,
    choices=RESIDUAL_REGISTRY.keys(),
    default="standard",
    help="The style of residual connection to use",
)
architecture_group.add_argument(
    "--compression-type",
    type=str,
    choices=COMPRESSION_REGISTRY.keys(),
    default="none",
    help="The type of sequence compression to use",
)
architecture_group.add_argument(
    "--sorting-type",
    type=str,
    choices=SORTING_REGISTRY.keys(),
    default="none",
    help="The type of feature sorting to use",
)
architecture_group.add_argument(
    "--activation",
    type=str,
    choices=ACTIVATION_REGISTRY.keys(),
    default="mish",
    help="The primary activation function to use",
)
architecture_group.add_argument(
    "--norm-type",
    type=str,
    choices=NORMALIZATION_REGISTRY.keys(),
    default="rms_norm",
    help="The type of normalization to use",
)
architecture_group.add_argument(
    "--head-type",
    type=str,
    choices=HEAD_REGISTRY.keys(),
    default="forward",
    help="The type of language modeling head to use",
)
architecture_group.add_argument(
    "--target-batch-size",
    type=int,
    default=256,
    help="The actual batch size to use, including accumulation steps",
)
architecture_group.add_argument(
    "--block-size",
    type=int,
    default=512,
    help="The base sequence length to train with",
)
architecture_group.add_argument(
    "--vocab-size",
    type=int,
    choices=[1024, 2048, 4096, 8192, 16384, 32768, 65536],
    default=16384,
    help="The absolute vocab size to use, though some architectures might scale it differently",
)
architecture_group.add_argument(
    "--depth",
    type=int,
    default=2,
    help="The max number of experts to route through",
)
architecture_group.add_argument(
    "--num-experts",
    type=int,
    default=None,
    help="Number of experts to host (defaults to depth)",
)
architecture_group.add_argument(
    "--num-smear",
    type=int,
    default=3,
    help="Number of SMEAR expert copies for GRU/recurrent blocks (default: 3)",
)
architecture_group.add_argument(
    "--hidden-size",
    type=int,
    default=256,
    help="The size of the model's hidden dimensions",
)
architecture_group.add_argument(
    "--embed-size",
    type=int,
    default=192,
    help="The size of the model's embedding dimension (if applicable)",
)
architecture_group.add_argument(
    "--dropout",
    type=int,
    default=0.1,
    help="The percentage of neurons to drop-out during training",
)
architecture_group.add_argument(
    "--num-heads",
    type=lambda x: (
        x
        if ":" in x
        and len(parts := x.split(":")) == 2
        and all(p.isdigit() for p in parts)
        and all(int(p) > 0 for p in parts)
        else (_ for _ in ()).throw(
            argparse.ArgumentTypeError(
                f"'{x}' is not in format 'X:Y', where X and Y are positive integers"
            )
        )
    ),
    default="4:2",
    help="The ratio of heads to queries per-head. (example: '4:2' is equal to 3 heads, with 2 queries per head)",
)
architecture_group.add_argument(
    "--head-size",
    type=int,
    default=None,
    help="Specify the inner head dimension",
)
architecture_group.add_argument(
    "--k-heads",
    type=int,
    default=None,
    help="A sparse MoE, controlling the number of heads to sample. Should be smaller than num_heads to enable.",
)
architecture_group.add_argument(
    "--kv-rank",
    type=int,
    default=None,
    help="Set this value to factorize key/value projections, making them low-rank. A value of 1 is lowest.",
)
architecture_group.add_argument(
    "--linear",
    action="store_true",
    default=False,
    help="Use a Linear (O(n)) attention mechanism",
)
architecture_group.add_argument(
    "--differential",
    action="store_true",
    default=False,
    help="Use a Differential Attention mechanism",
)
architecture_group.add_argument(
    "--stickbreaking",
    action="store_true",
    default=False,
    help="Use a Stickbreaking Attention mechanism",
)
architecture_group.add_argument(
    "--memory",
    action="store_true",
    default=False,
    help="Use a long-term episodic memory module",
)
architecture_group.add_argument(
    "--mla",
    action="store_true",
    default=False,
    help="Use Multi-Head Latent Attention (MLA)",
)
architecture_group.add_argument(
    "--mta",
    action="store_true",
    default=False,
    help="Use Multi-Token Attention (MTA)",
)
architecture_group.add_argument(
    "--mega",
    action="store_true",
    default=False,
    help="Equip the attention mechanism with exponentially-moving average-based gating",
)
architecture_group.add_argument(
    "--gated",
    action="store_true",
    default=False,
    help="Add a gating network to attention outputs",
)
architecture_group.add_argument(
    "--evolve",
    action="store_true",
    default=False,
    help="Use a genomic bottleneck",
)
architecture_group.add_argument(
    "--scaled",
    action="store_true",
    default=False,
    help="Scale the output of each layer by the inverse square root of its depth",
)
architecture_group.add_argument(
    "--bidirectional",
    action="store_true",
    default=False,
    help="Enable bidirectional language modeling (forward and backward prediction)",
)
architecture_group.add_argument(
    "--tie-weights",
    action="store_true",
    default=False,
    help="Tie embedding and output projection weights to reduce parameters",
)
# optimization
optimization_group.add_argument(
    "--optimizer",
    type=str,
    choices=OPTIMIZER_PROFILES.keys(),
    default="Lion",
    help="The optimizer profile to use",
)
optimization_group.add_argument(
    "--loss-func",
    type=str,
    choices=LOSS_REGISTRY.keys(),
    default="cross_entropy",
    help="The loss function to use",
)
optimization_group.add_argument(
    "--strategy",
    type=str,
    choices=STRATEGIES_REGISTRY.keys(),
    default="naive",
    help="The multitask objective strategy to use for loss combination",
)
optimization_group.add_argument(
    "--trac",
    action="store_true",
    default=False,
    help="Wrap the optimizer in TRAC, which can mitigate the loss of plasticity over time",
)
optimization_group.add_argument(
    "--ortho",
    action="store_true",
    default=False,
    help="Wrap the optimizer in OrthoGrad, projecting gradients to be orthogonal to parameters",
)
optimization_group.add_argument(
    "--lookahead",
    action="store_true",
    default=False,
    help="Wrap the optimizer in Lookahead",
)
optimization_group.add_argument(
    "--fixed-schedule",
    action="store_true",
    default=False,
    help="Use a fixed (constant) learning rate schedule",
)
optimization_group.add_argument(
    "--schedule-free",
    action="store_true",
    default=False,
    help="Use the Schedule-Free optimizer wrapper",
)
# networking
# data
data_group.add_argument(
    "--data-path",
    type=str,
    nargs="+",
    default=None,
    help="Paths to a directory of files to use as training data",
)
data_group.add_argument(
    "--pile",
    action="store_true",
    default=False,
    help="Train exclusively on the minipile challenge dataset",
)
data_group.add_argument(
    "--phi",
    action="store_true",
    default=False,
    help="Supplement training with a mix of expert data",
)
data_group.add_argument(
    "--no-source",
    action="store_true",
    default=False,
    help="Disable training on the model's own source code",
)
data_group.add_argument(
    "--rl-type",
    type=str,
    default=None,
    choices=RL_POLICIES_REGISTRY.keys(),
    help="Enable reinforcement learning with specified algorithm. "
    "Note: Current GRPO implementation uses static dataset rewards (not true RL). "
    "True RL with generation will be added in a future update.",
)
# other
other_group.add_argument(
    "--seed",
    type=int,
    default=int(65536 * (2 * math.acos(1 - random.random()) / math.pi) ** 6.66),
    help="Global seed used for reproducibility",
)
other_group.add_argument(
    "--meta",
    type=str,
    action="append",
    default=[],
    help="Append keywords to a list at 'config.meta'. Used for model development. You probably don't need this.",
)
other_group.add_argument(
    "--no-dashboard",
    action="store_true",
    default=False,
    help="Disable the terminal dashboard",
)
other_group.add_argument(
    "--quiet",
    action="store_true",
    default=False,
    help="Suppress text generation in the terminal",
)
# Only add --dev if it wasn't already added by environment loading
if not any("--dev" in action.option_strings for action in parser._actions):
    other_group.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Use fewer resources (3 layers, smaller datasets, etc), always start from a new model (i.e. force '--reset'), and never conflict/remove existing, saved models. Can be used simultaneously alongside an active, running 'live' model.",
    )
other_group.add_argument(
    "--eval-every",
    type=int,
    default=None,
    help="Run partial evaluation every N validation intervals",
)
other_group.add_argument(
    "--eval-tasks",
    type=str,
    default="helm|hellaswag|2|1,lighteval|glue:cola|2|1,lighteval|coqa|2|1",
    help="Run a subset of evaluation tests after each validation step. This can be slow.",
)
other_group.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Print debug logs to the terminal",
)
other_group.add_argument(
    "--reset",
    action="store_true",
    default=False,
    help="Reset the checkpoint",
)


# Experiment loading functionality
# Note: experiment_configs and environment_configs are already loaded above

# Destructure CLI arguments
args = parser.parse_args()

# Store original parsed values to detect what was explicitly provided
explicitly_provided = set()
for arg in sys.argv[1:]:  # Skip script name
    if arg.startswith("--") and "=" in arg:
        # Handle --arg=value format
        key = arg.split("=")[0][2:]
        explicitly_provided.add(key)
    elif arg.startswith("--"):
        # Handle --arg value format
        key = arg[2:]
        explicitly_provided.add(key)

# Apply experiment defaults if any experiment flags are set
if experiment_configs:
    for experiment_name, config in experiment_configs.items():
        if getattr(args, experiment_name.replace("-", "_"), False):
            # Apply experiment defaults (but don't override user-provided values)
            for key, value in config.items():
                # Convert key to argument name format (replace - with _)
                attr_name = key.replace("-", "_")

                # Check if this argument was explicitly provided by the user
                if (
                    key in explicitly_provided
                    or key.replace("_", "-") in explicitly_provided
                ):
                    continue  # User override takes precedence

                # Apply the experiment default
                # Always set the value, even if the attribute doesn't exist yet
                setattr(args, attr_name, value)

# Apply environment overrides LAST (highest priority) - they override everything
# Also check for mutual exclusivity and handle --dev backward compatibility
from praxis.environments import EnvironmentFeatures

active_environment = None
environment_features = {}

# Check for environment flags and apply them
for env_name, env_config in environment_configs.items():
    if getattr(args, env_name.replace("-", "_"), False):
        if active_environment and active_environment != env_name:
            raise ValueError(
                f"Cannot use multiple environments simultaneously: --{active_environment} and --{env_name}"
            )
        active_environment = env_name

# Apply the active environment's configuration if one is set
if active_environment and active_environment in environment_configs:
    env_config = environment_configs[active_environment]

    # Apply overrides (these have highest priority, overwrite everything)
    for key, value in env_config["overrides"].items():
        attr_name = key.replace("-", "_")
        setattr(args, attr_name, value)

    # Store features for global access
    environment_features = env_config["features"]

# Set environment features globally for use throughout the codebase
EnvironmentFeatures.set_from_environment(environment_features, active_environment)

# Now check integration conditions based on parsed args (with environment overrides applied)
for integration_manifest in integrations:
    integration_loader.load_integration(integration_manifest, args, verbose=False)

# Print clean summary of integrations
integration_loader.print_summary()

# Export the integration loader for use in other modules


def apply_defaults_and_parse(defaults_dict):
    """
    Apply custom default values to the parser and re-parse arguments.

    This allows scripts to override defaults while maintaining full CLI compatibility.
    The logged command will show the actual script used, but the hash will be computed
    from the effective configuration for consistency.

    Args:
        defaults_dict: Dictionary mapping argument names (with underscores) to default values

    Returns:
        tuple: (parsed_args, hash_computed_from_effective_config)
    """
    global args

    # Store original command for logging
    original_command = sys.argv[:]

    # Apply custom defaults to parser
    for action in parser._actions:
        if hasattr(action, "dest") and action.dest in defaults_dict:
            action.default = defaults_dict[action.dest]

    # Re-parse arguments with new defaults
    args = parser.parse_args()

    # Also apply experiment defaults if any experiment flags are set
    if experiment_configs:
        for experiment_name, config in experiment_configs.items():
            if getattr(args, experiment_name.replace("-", "_"), False):
                # Apply experiment defaults (but don't override user-provided values)
                for key, value in config.items():
                    # Convert key to argument name format (replace - with _)
                    attr_name = key.replace("-", "_")

                    # Check if this argument was explicitly provided by the user
                    if attr_name in sys.argv or f"--{key}" in sys.argv:
                        continue  # User override takes precedence

                    # Apply the experiment default
                    # Always set the value, even if the attribute doesn't exist yet
                    setattr(args, attr_name, value)

    # Apply environment overrides in apply_defaults_and_parse too
    # This handles environments when using custom scripts
    from praxis.environments import EnvironmentFeatures

    active_env = None
    env_features = {}

    # Check for --dev backward compatibility
    if getattr(args, "dev", False) and "dev" in environment_configs:
        active_env = "dev"
        env_features = environment_configs["dev"]["features"]
        for key, value in environment_configs["dev"]["overrides"].items():
            setattr(args, key.replace("-", "_"), value)

    # Apply any other active environment
    for env_name in environment_configs:
        if getattr(args, env_name.replace("-", "_"), False) and env_name != "dev":
            if active_env:
                raise ValueError(
                    f"Cannot use multiple environments: --{active_env} and --{env_name}"
                )
            active_env = env_name
            env_features = environment_configs[env_name]["features"]
            for key, value in environment_configs[env_name]["overrides"].items():
                setattr(args, key.replace("-", "_"), value)

    EnvironmentFeatures.set_from_environment(env_features, active_env)

    # Re-evaluate integration conditions with new args
    for integration_manifest in integrations:
        integration_loader.load_integration(integration_manifest, args, verbose=False)

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
            if i + 1 < len(equivalent_args) and not equivalent_args[i + 1].startswith(
                "-"
            ):
                filtered_defaults.append(equivalent_args[i + 1])
                i += 2
            else:
                i += 1
        else:
            if i + 1 < len(equivalent_args) and not equivalent_args[i + 1].startswith(
                "-"
            ):
                i += 2
            else:
                i += 1

    # Combine for hash computation
    hash_args = filtered_defaults + user_args

    # Compute hash from effective configuration
    effective_hash = _compute_args_hash(hash_args)

    # Set up custom logging that shows original command but uses computed hash
    global log_command
    original_log_command = log_command

    def custom_log_command(exclude_from_hash=None):
        if exclude_from_hash is None:
            exclude_from_hash = DEFAULT_EXCLUDE_FROM_HASH

        # Log the original command with computed hash
        script_name = os.path.basename(original_command[0])
        args_list = original_command[1:]
        displayed_command = f"python {script_name} {' '.join(args_list)}"

        # If this is run_alpha.py, also construct the full equivalent run.py command
        log_entry_command = displayed_command
        if script_name == "run_alpha.py":
            # Reconstruct the equivalent run.py command with all the applied defaults
            full_args = []
            for key, value in vars(args).items():
                # Convert back to command line format
                arg_name = f"--{key.replace('_', '-')}"
                if isinstance(value, bool):
                    if value:  # Only add flag if it's True
                        full_args.append(arg_name)
                elif value is not None:
                    full_args.append(arg_name)
                    full_args.append(str(value))

            full_run_command = f"python run.py {' '.join(full_args)}"
            log_entry_command = f'"{displayed_command}" | "{full_run_command}"'

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        truncated_hash = effective_hash[:9]
        new_entry = f"{timestamp} | {truncated_hash} | {log_entry_command}\n"

        # Write to history.log
        log_file = "history.log"
        existing_content = ""
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                existing_content = f.read()

        with open(log_file, "w") as f:
            f.write(new_entry + existing_content)

        return displayed_command, effective_hash, truncated_hash

    # Replace log_command function
    log_command = custom_log_command

    return args, effective_hash


def _compute_args_hash(args_list, exclude_from_hash=None):
    """Compute hash from argument list (internal helper)."""
    if exclude_from_hash is None:
        exclude_from_hash = DEFAULT_EXCLUDE_FROM_HASH
    arg_dict = {}
    i = 0
    while i < len(args_list):
        if args_list[i].startswith("-"):
            arg_name = args_list[i]
            if i + 1 < len(args_list) and not args_list[i + 1].startswith("-"):
                if arg_name not in exclude_from_hash:
                    arg_dict[arg_name] = args_list[i + 1]
                i += 2
            else:
                if arg_name not in exclude_from_hash:
                    arg_dict[arg_name] = True
                i += 1
        else:
            pos_arg_name = f"_pos_{i}"
            if pos_arg_name not in exclude_from_hash:
                arg_dict[pos_arg_name] = args_list[i]
            i += 1

    sorted_args = dict(sorted(arg_dict.items()))
    args_json = json.dumps(sorted_args, sort_keys=True)
    hash_object = hashlib.sha256(args_json.encode())
    return hash_object.hexdigest()


def create_praxis_config(args=None, tokenizer=None):
    """
    Create a PraxisConfig object from CLI arguments.
    Automatically maps CLI arguments to PraxisConfig parameters based on the config's signature.

    Args:
        args: Parsed arguments object (defaults to global args)
        tokenizer: Optional tokenizer for token IDs (if not provided, uses defaults)

    Returns:
        PraxisConfig: Configuration object ready for model initialization
    """
    import inspect

    from praxis import PraxisConfig

    if args is None:
        args = globals()["args"]

    # Get PraxisConfig's __init__ parameters to know what it accepts
    config_signature = inspect.signature(PraxisConfig.__init__)
    valid_config_params = set(config_signature.parameters.keys()) - {"self", "kwargs"}

    # Extract and transform arguments
    config_kwargs = {}

    # Mapping of CLI argument names to config parameter names (for renamed parameters)
    arg_to_config_mapping = {
        "expert_type": "expert",
        "encoding_type": "encoding",
    }

    # Process all arguments from CLI
    for arg_name in vars(args):
        arg_value = getattr(args, arg_name)

        # Skip None values
        if arg_value is None:
            continue

        # Check if this arg needs to be mapped to a different config param name
        config_param = arg_to_config_mapping.get(arg_name, arg_name)

        # Only include parameters that PraxisConfig actually accepts
        if config_param in valid_config_params:
            config_kwargs[config_param] = arg_value

    # Special transformations that require custom logic

    # num_heads and num_queries from "num_heads:num_queries" format
    if hasattr(args, "num_heads") and args.num_heads:
        parts = args.num_heads.split(":")
        config_kwargs["num_heads"] = int(parts[0])
        config_kwargs["num_queries"] = (
            int(parts[1]) if len(parts) > 1 else int(parts[0])
        )

    # num_experts defaults to depth if not specified
    # (Environment overrides will have already been applied if active)
    if "num_experts" not in config_kwargs or config_kwargs.get("num_experts") is None:
        config_kwargs["num_experts"] = config_kwargs.get("depth", 2)

    # Handle byte_latent encoding
    byte_latent = hasattr(args, "encoder_type") and args.encoder_type == "byte_latent"
    if byte_latent and "byte_latent" in valid_config_params:
        config_kwargs["byte_latent"] = True

    # Handle max_length based on block_size
    if hasattr(args, "block_size") and "max_length" in valid_config_params:
        block_size = args.block_size
        # Adjust for byte_latent if needed
        if byte_latent:
            block_size = block_size * 8
        config_kwargs["max_length"] = block_size * 8

    # Handle tokenizer IDs if tokenizer is provided
    if tokenizer is not None:
        token_id_mappings = [
            ("pad_token_id", "pad_token_id"),
            ("bos_token_id", "bos_token_id"),
            ("eos_token_id", "eos_token_id"),
            ("sep_token_id", "sep_token_id"),
        ]
        for tokenizer_attr, config_param in token_id_mappings:
            if (
                hasattr(tokenizer, tokenizer_attr)
                and config_param in valid_config_params
            ):
                config_kwargs[config_param] = getattr(tokenizer, tokenizer_attr)

    # Handle optimizer configuration
    if hasattr(args, "optimizer") and args.optimizer:
        from praxis.optimizers import get_optimizer_profile

        # Check for schedule-related flags
        disable_schedule = any(
            [
                getattr(args, "fixed_schedule", False),
                getattr(args, "schedule_free", False),
            ]
        )

        optimizer_config, _ = get_optimizer_profile(args.optimizer, disable_schedule)

        if "optimizer_config" in valid_config_params:
            config_kwargs["optimizer_config"] = optimizer_config

        # Add optimizer wrappers if they're valid config params
        if "optimizer_wrappers" in valid_config_params:
            config_kwargs["optimizer_wrappers"] = {
                "trac": getattr(args, "trac", False),
                "ortho": getattr(args, "ortho", False),
                "lookahead": getattr(args, "lookahead", False),
                "schedule_free": getattr(args, "schedule_free", False),
            }

    # Filter out any kwargs that aren't valid for PraxisConfig
    # This ensures we only pass parameters the config actually accepts
    filtered_kwargs = {
        k: v for k, v in config_kwargs.items() if k in valid_config_params
    }

    return PraxisConfig(**filtered_kwargs)


def get_processed_args(args=None):
    """
    Get a dictionary of processed arguments suitable for use in main.py.
    This handles all the preprocessing that was previously done via globals().update().

    Args:
        args: Parsed arguments object (defaults to global args)

    Returns:
        dict: Processed arguments ready for use
    """
    if args is None:
        args = globals()["args"]

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

    # Environment overrides will have already been applied by this point
    # No need for hardcoded dev logic here

    return processed


def get_cli_args():
    return args


def log_command(exclude_from_hash=None):
    """
    Logs the current command line execution to history.log in the root directory.
    New commands are added to the top of the file.
    Also computes and stores a hash of the arguments that is order-independent.

    Args:
        exclude_from_hash (list, optional): List of argument names to exclude from hashing.
            Example: ['--verbose', '--log-level', '--output']
            Defaults to DEFAULT_EXCLUDE_FROM_HASH if None.

    Returns:
        tuple: (full_command, args_hash)
    """
    # Use default exclude list if None provided
    if exclude_from_hash is None:
        exclude_from_hash = DEFAULT_EXCLUDE_FROM_HASH

    # Construct the command
    script_name = os.path.basename(sys.argv[0])
    args = sys.argv[1:]
    full_command = f"python {script_name} {' '.join(args)}"

    # Create a normalized representation of arguments for hashing
    # We'll create a dictionary of argument names and values
    arg_dict = {}
    i = 0
    while i < len(args):
        if args[i].startswith("-"):
            # This is an argument name
            arg_name = args[i]

            # Check if next item is a value or another flag
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                # This is a value
                # Only add to arg_dict if not in exclude list
                if arg_name not in exclude_from_hash:
                    arg_dict[arg_name] = args[i + 1]
                i += 2
            else:
                # This is a flag without value
                # Only add to arg_dict if not in exclude list
                if arg_name not in exclude_from_hash:
                    arg_dict[arg_name] = True
                i += 1
        else:
            # This is a positional argument
            # For positional args, we need to be careful with exclusions
            # We'll use an index that accounts for excluded positional args
            pos_arg_name = f"_pos_{i}"
            # Always add positional arguments unless specifically excluded
            if pos_arg_name not in exclude_from_hash:
                arg_dict[pos_arg_name] = args[i]
            i += 1

    # Sort the dictionary by keys for consistent order
    sorted_args = dict(sorted(arg_dict.items()))

    # Create a JSON string for hashing (ensures consistent formatting)
    args_json = json.dumps(sorted_args, sort_keys=True)

    # Generate hash
    hash_object = hashlib.sha256(args_json.encode())
    args_hash = hash_object.hexdigest()

    # Format log entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    truncate_to = 9
    truncated_hash = args_hash[:truncate_to]
    new_entry = f'{timestamp} | {truncated_hash} | "{full_command}"\n'

    # Get the path for history.log in root directory
    log_file = "history.log"

    # Read existing content (if any)
    existing_content = ""
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            existing_content = f.read()

    # Write new entry followed by existing content
    with open(log_file, "w") as f:
        f.write(new_entry + existing_content)

    return full_command, args_hash, truncated_hash


# Export the integration loader for use in run.py
__all__ = [
    "integration_loader",
    "args",
    "create_praxis_config",
    "get_processed_args",
    "get_cli_args",
]
