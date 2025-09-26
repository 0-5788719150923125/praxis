"""Hash computation for CLI arguments."""

import hashlib
import json

# Define the default list of arguments to exclude from hash computation
# These are typically runtime/debugging flags that don't affect model architecture
DEFAULT_EXCLUDE_FROM_HASH = [
    "--reset",
    "--preserve",
    "--list-runs",
    "--debug",
    "--ngrok",
    "--wandb",
    "--no-dashboard",
]


def compute_args_hash(args_list, exclude_from_hash=None):
    """
    Compute a deterministic hash from a list of command-line arguments.

    Args:
        args_list: List of command-line arguments (without script name)
        exclude_from_hash: List of argument names to exclude from hashing

    Returns:
        str: SHA256 hash of the normalized arguments
    """
    if exclude_from_hash is None:
        exclude_from_hash = DEFAULT_EXCLUDE_FROM_HASH

    arg_dict = {}
    i = 0
    while i < len(args_list):
        if args_list[i].startswith("-"):
            # This is an argument name
            arg_name = args_list[i]

            # Check if next item is a value or another flag
            if i + 1 < len(args_list) and not args_list[i + 1].startswith("-"):
                # This is a value
                if arg_name not in exclude_from_hash:
                    arg_dict[arg_name] = args_list[i + 1]
                i += 2
            else:
                # This is a flag without value
                if arg_name not in exclude_from_hash:
                    arg_dict[arg_name] = True
                i += 1
        else:
            # This is a positional argument
            pos_arg_name = f"_pos_{i}"
            if pos_arg_name not in exclude_from_hash:
                arg_dict[pos_arg_name] = args_list[i]
            i += 1

    # Sort the dictionary by keys for consistent order
    sorted_args = dict(sorted(arg_dict.items()))

    # Create a JSON string for hashing (ensures consistent formatting)
    args_json = json.dumps(sorted_args, sort_keys=True)

    # Generate hash
    hash_object = hashlib.sha256(args_json.encode())
    return hash_object.hexdigest()
