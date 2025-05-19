import argparse
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime

from optimizers import OPTIMIZER_PROFILES
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
    LOSS_REGISTRY,
    RESIDUAL_REGISTRY,
    ROUTER_REGISTRY,
    SORTING_REGISTRY,
    STRATEGIES_REGISTRY,
)


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

# hardware
hardware_group.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to use",
)
hardware_group.add_argument(
    "--host_name",
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
    "--batch_size",
    type=int,
    default=1,
    help="Batch size to use for training",
)
hardware_group.add_argument(
    "--checkpoint_every",
    type=int,
    default=None,
    help="Apply gradient checkpointing every X layers",
)
# storage
persistence_group.add_argument(
    "--cache_dir",
    type=str,
    default="data",
    help="Paths to a directory where artifacts will be saved",
)
# architecture
architecture_group.add_argument(
    "--encoder_type",
    type=str,
    choices=list(ENCODER_REGISTRY.keys()),
    default=None,
    help="Encoder module to use",
)
architecture_group.add_argument(
    "--decoder_type",
    type=str,
    choices=list(DECODER_REGISTRY.keys()),
    default="sequential",
    help="How to process layers in the decoder",
)
architecture_group.add_argument(
    "--block_type",
    type=str,
    choices=BLOCK_REGISTRY.keys(),
    default="transformer",
    help="The type of block to use for every intermediate decoder layer",
)
architecture_group.add_argument(
    "--expert_type",
    type=str,
    choices=EXPERT_REGISTRY.keys(),
    default="glu",
    help="The module to use for feedforward networks",
)
architecture_group.add_argument(
    "--attention_type",
    type=str,
    choices=ATTENTION_REGISTRY.keys(),
    default="standard",
    help="The base attention implementation to use",
)
architecture_group.add_argument(
    "--encoding_type",
    type=str,
    choices=ENCODING_REGISTRY.keys(),
    default="rope",
    help="The positional encoding to use for sequence length extrapolation",
)
architecture_group.add_argument(
    "--controller_type",
    type=str,
    choices=CONTROLLER_REGISTRY.keys(),
    default="base",
    help="Various methods used to route inputs through experts in the decoder",
)
architecture_group.add_argument(
    "--router_type",
    type=str,
    choices=ROUTER_REGISTRY.keys(),
    default=None,
    help="How to route tokens at every layer",
)
architecture_group.add_argument(
    "--residual_type",
    type=str,
    choices=RESIDUAL_REGISTRY.keys(),
    default="standard",
    help="The style of residual connection to use",
)
architecture_group.add_argument(
    "--compression_type",
    type=str,
    choices=COMPRESSION_REGISTRY.keys(),
    default="none",
    help="The type of sequence compression to use",
)
architecture_group.add_argument(
    "--sorting_type",
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
    "--target_batch_size",
    type=int,
    default=256,
    help="The actual batch size to use, including accumulation steps",
)
architecture_group.add_argument(
    "--block_size",
    type=int,
    default=512,
    help="The base sequence length to train with",
)
architecture_group.add_argument(
    "--vocab_size",
    type=int,
    choices=[1024, 2048, 4096, 8192, 16384, 32768, 65536],
    default=16384,
    help="The absolute vocab size to use, though some architectures might scale it differently",
)
architecture_group.add_argument(
    "--depth",
    type=int,
    default=9,
    help="The max number of experts to route through",
)
architecture_group.add_argument(
    "--num_experts",
    type=int,
    default=False,
    help="Number of experts to host (defaults to depth)",
)
architecture_group.add_argument(
    "--hidden_size",
    type=int,
    default=256,
    help="The size of the model's hidden dimensions",
)
architecture_group.add_argument(
    "--embed_size",
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
    "--num_heads",
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
    "--k_heads",
    type=int,
    default=None,
    help="A sparse MoE, controlling the number of heads to sample. Should be smaller than num_heads to enable.",
)
architecture_group.add_argument(
    "--kv_rank",
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
# optimization
optimization_group.add_argument(
    "--optimizer",
    type=str,
    choices=OPTIMIZER_PROFILES.keys(),
    default="Lion",
    help="The optimizer profile to use",
)
optimization_group.add_argument(
    "--loss_func",
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
    "--fixed_schedule",
    action="store_true",
    default=False,
    help="Use a fixed (constant) learning rate schedule",
)
optimization_group.add_argument(
    "--schedule_free",
    action="store_true",
    default=False,
    help="Use the Schedule-Free optimizer wrapper",
)
# networking
networking_group.add_argument(
    "--hivemind",
    action="store_true",
    default=False,
    help="Connect your node to the Hivemind swarm",
)
networking_group.add_argument(
    "--initial_peers",
    nargs="*",
    default=[],
    help="Provide a list of Hivemind bootstrap peers",
)
# data
data_group.add_argument(
    "--data_path",
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
    "--gun",
    action="store_true",
    default=False,
    help="Supplement training with chat data from https://src.eco",
)
data_group.add_argument(
    "--no_source",
    action="store_true",
    default=False,
    help="Disable training on the model's own source code",
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
    "--wandb",
    action="store_true",
    default=False,
    help="Log metrics to Weights and Biases (https://wandb.ai)",
)
other_group.add_argument(
    "--wandb_run_name",
    type=str,
    default=None,
    help="Custom name for the W&B run (default: auto-generated)",
)
other_group.add_argument(
    "--no_dashboard",
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
other_group.add_argument(
    "--dev",
    action="store_true",
    default=False,
    help="Use fewer resources (3 layers, smaller datasets, etc), always start from a new model (i.e. force '--reset'), and never conflict/remove existing, saved models. Can be used simultaneously alongside an active, running 'live' model.",
)
other_group.add_argument(
    "--eval_every",
    type=int,
    default=None,
    help="Run partial evaluation every N validation intervals",
)
other_group.add_argument(
    "--eval_tasks",
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

# Destructure CLI arguments
args = parser.parse_args()


def get_cli_args():
    return args


def log_command(exclude_from_hash=["--reset", "--debug"]):
    """
    Logs the current command line execution to history.log in the root directory.
    New commands are added to the top of the file.
    Also computes and stores a hash of the arguments that is order-independent.

    Args:
        exclude_from_hash (list, optional): List of argument names to exclude from hashing.
            Example: ['--verbose', '--log-level', '--output']

    Returns:
        tuple: (full_command, args_hash)
    """
    # Default to empty list if None
    if exclude_from_hash is None:
        exclude_from_hash = []

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
    new_entry = f'{timestamp} | {args_hash[:truncate_to]} | "{full_command}"\n'

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

    return full_command, args_hash
