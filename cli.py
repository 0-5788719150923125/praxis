import argparse
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
    ENCODING_REGISTRY,
    EXPERT_REGISTRY,
    LOSS_REGISTRY,
)


# User args, accepted via CLI
class CustomHelpFormatter(argparse.HelpFormatter):
    def _format_usage(self, usage, actions, groups, prefix):
        return ""  # This effectively removes the usage section


parser = argparse.ArgumentParser(
    description="User-supplied arguments to this script.",
    formatter_class=CustomHelpFormatter,
)
parser.add_argument(
    "--seed",
    type=int,
    default=int(65536 * (2 * math.acos(1 - random.random()) / math.pi) ** 6.66),
    help="Global seed",
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to use",
)
parser.add_argument(
    "--host_name",
    type=str,
    default="localhost",
    help="Serve the local API at this CNAME",
)
parser.add_argument(
    "--port",
    type=int,
    default=2100,
    help="Serve the local API at this port",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size to use for training",
)
parser.add_argument(
    "--target_batch_size",
    type=int,
    default=64,
    help="The actual batch size to use, including accumulation steps",
)
parser.add_argument(
    "--block_size",
    type=int,
    default=512,
    help="The base sequence length to train with",
)
parser.add_argument(
    "--vocab_size",
    type=int,
    choices=[1024, 2048, 4096, 8192, 16384, 32768, 65536],
    default=8192,
    help="The absolute vocab size to use, though some architectures might scale it differently",
)
parser.add_argument(
    "--depth",
    type=int,
    default=7,
    help="The max number of experts to route through",
)
parser.add_argument(
    "--num_experts",
    type=int,
    default=False,
    help="Number of experts to host (defaults to depth)",
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=256,
    help="The size of the model's hidden dimensions",
)
parser.add_argument(
    "--embed_size",
    type=int,
    default=192,
    help="The size of the model's embedding dimension (if applicable)",
)
parser.add_argument(
    "--dropout",
    type=int,
    default=0.1,
    help="The percentage of neurons to drop-out during training",
)
parser.add_argument(
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
parser.add_argument(
    "--k_heads",
    type=int,
    default=None,
    help="A sparse MoE, controlling the number of heads to sample. Should be smaller than num_heads to enable.",
)
parser.add_argument(
    "--kv_rank",
    type=int,
    default=None,
    help="Set this value to factorize key/value projections, making them low-rank. A value of 1 is lowest.",
)
parser.add_argument(
    "--activation",
    type=str,
    choices=ACTIVATION_REGISTRY.keys(),
    default="mish",
    help="The primary activation function to use",
)
parser.add_argument(
    "--attention_type",
    type=str,
    choices=ATTENTION_REGISTRY.keys(),
    default="standard",
    help="The base attention implementation to use",
)
parser.add_argument(
    "--checkpoint_every",
    type=int,
    default=None,
    help="Apply gradient checkpointing every X layers",
)
parser.add_argument(
    "--data_path",
    type=str,
    nargs="+",
    default=None,
    help="Paths to a directory of files to use as training data",
)
parser.add_argument(
    "--cache_dir",
    type=str,
    default="data",
    help="Paths to a directory where artifacts will be saved",
)
parser.add_argument(
    "--no_dashboard",
    action="store_true",
    default=False,
    help="Disable the terminal dashboard",
)
parser.add_argument(
    "--no_schedule",
    action="store_true",
    default=False,
    help="Disable the learning rate scheduler",
)
parser.add_argument(
    "--schedule_free",
    action="store_true",
    default=False,
    help="Use the Schedule-Free optimizer wrapper",
)
parser.add_argument(
    "--wandb",
    action="store_true",
    default=False,
    help="Log metrics to Weights and Biases (https://wandb.ai)",
)
parser.add_argument(
    "--optimizer",
    type=str,
    choices=OPTIMIZER_PROFILES.keys(),
    default="Lion",
    help="The optimizer profile to use",
)
parser.add_argument(
    "--trac",
    action="store_true",
    default=False,
    help="Wrap the optimizer in TRAC, which can mitigate the loss of plasticity over time",
)
parser.add_argument(
    "--ortho",
    action="store_true",
    default=False,
    help="Wrap the optimizer in OrthoGrad, projecting gradients to be orthogonal to parameters",
)
parser.add_argument(
    "--lookahead",
    action="store_true",
    default=False,
    help="Wrap the optimizer in Lookahead",
)
parser.add_argument(
    "--loss_func",
    type=str,
    choices=LOSS_REGISTRY.keys(),
    default="cross_entropy",
    help="The loss function to use",
)
parser.add_argument(
    "--block_type",
    type=str,
    choices=BLOCK_REGISTRY.keys(),
    default="transformer",
    help="The type of block to use for every intermediate decoder layer",
)
parser.add_argument(
    "--expert_type",
    type=str,
    choices=EXPERT_REGISTRY.keys(),
    default="glu",
    help="The module to use for feedforward networks",
)
parser.add_argument(
    "--encoding_type",
    type=str,
    choices=ENCODING_REGISTRY.keys(),
    default="rope",
    help="The positional encoding to use for sequence length extrapolation",
)
parser.add_argument(
    "--dense",
    action="store_true",
    default=True,
    help="Run as a dense model",
)
parser.add_argument(
    "--sparse",
    action="store_true",
    default=False,
    help="Run as a sparse model",
)
parser.add_argument(
    "--shuffle",
    action="store_true",
    default=False,
    help="Shuffle layers at every forward pass",
)
parser.add_argument(
    "--autopilot",
    action="store_true",
    default=False,
    help="Allow the model to discover and route through experts in a optimal fashion",
)
parser.add_argument(
    "--graph",
    action="store_true",
    default=False,
    help="Use graph-based routing through experts/layers",
)
parser.add_argument(
    "--router",
    action="store_true",
    default=False,
    help="Use a simple router to select optimal experts/layers",
)
parser.add_argument(
    "--compression",
    action="store_true",
    default=False,
    help="Compress sequence length by a factor of 2",
)
parser.add_argument(
    "--linear",
    action="store_true",
    default=False,
    help="Use a Linear (O(n)) attention mechanism",
)
parser.add_argument(
    "--differential",
    action="store_true",
    default=False,
    help="Use a Differential Attention mechanism",
)
parser.add_argument(
    "--stickbreaking",
    action="store_true",
    default=False,
    help="Use a Stickbreaking Attention mechanism",
)
parser.add_argument(
    "--memory",
    action="store_true",
    default=False,
    help="Use a long-term episodic memory module",
)
parser.add_argument(
    "--mega",
    action="store_true",
    default=False,
    help="Equip the attention mechanism with exponentially-moving average-based gating",
)
parser.add_argument(
    "--gated",
    action="store_true",
    default=False,
    help="Add a gating network to attention outputs",
)
parser.add_argument(
    "--evolve",
    action="store_true",
    default=False,
    help="Use a genomic bottleneck",
)
parser.add_argument(
    "--byte_latent",
    action="store_true",
    default=False,
    help="Use a Byte Latent Tokenizer (BLT)",
)
parser.add_argument(
    "--hyper",
    action="store_true",
    default=False,
    help="Replace residual connections with hyper-connections",
)
parser.add_argument(
    "--scaling",
    action="store_true",
    default=False,
    help="Scale the variance normalization inversely by the square root of layer depth",
)
parser.add_argument(
    "--hivemind",
    action="store_true",
    default=False,
    help="Connect your node to the Hivemind swarm",
)
parser.add_argument(
    "--initial_peers",
    nargs="*",
    default=[],
    help="Provide a list of Hivemind bootstrap peers",
)
parser.add_argument(
    "--pile",
    action="store_true",
    default=False,
    help="Train exclusively on the minipile challenge dataset",
)
parser.add_argument(
    "--phi",
    action="store_true",
    default=False,
    help="Supplement training with a mix of expert data",
)
parser.add_argument(
    "--gun",
    action="store_true",
    default=False,
    help="Supplement training with chat data from https://src.eco",
)
parser.add_argument(
    "--source",
    action="store_true",
    default=True,
    help="Train on the model's own source code",
)
parser.add_argument(
    "--quiet",
    action="store_true",
    default=False,
    help="Suppress text generation in the terminal",
)
parser.add_argument(
    "--use_cache",
    action="store_true",
    default=False,
    help="Use KV caching during inference, at the cost of text consistency in the dashboard (but not in the API)",
)
parser.add_argument(
    "--dev",
    action="store_true",
    default=False,
    help="Bootstrap faster (with 3 layers, a smaller dataset, etc.)",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Print debug logs to the terminal",
)
parser.add_argument(
    "--meta",
    type=str,
    action="append",
    default=[],
    help="Can be specified multiple times to build a list of meta flags",
)
parser.add_argument(
    "--reset",
    action="store_true",
    default=False,
    help="Reset the checkpoint",
)

# Destructure CLI arguments
args = parser.parse_args()


def get_cli_args():
    return args


def log_command():
    """
    Logs the current command line execution to history.log in the root directory.
    New commands are added to the top of the file.
    Returns the logged command string.
    """
    # Construct the command and log entry
    script_name = os.path.basename(sys.argv[0])
    full_command = f"python {script_name} {' '.join(sys.argv[1:])}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = f"{timestamp} | {full_command}\n"

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

    return full_command
