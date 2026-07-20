"""Hash computation for CLI arguments."""

import hashlib
import json

# Define the default list of arguments to exclude from hash computation
# These are typically runtime/debugging flags that don't affect model architecture
DEFAULT_EXCLUDE_FROM_HASH = [
    "--reset",
    "--reset-after",
    "--preserve",
    "--list-runs",
    "--debug",
    "--ngrok",
    "--wandb",
    "--no-dashboard",
    "--headless",
    "--quiet",
    "--no-checkpoints",
    "--num-nodes",
    "--node-rank",
    "--master-addr",
    "--master-port",
    "--infer-every",
    "--profile-memory",
    "--profile-memory-start",
    "--profile-memory-steps",
    "--profile-memory-max-entries",
]


def _integration_hash_exclusions():
    """Flags that loaded integrations asked to keep out of the run hash.

    Integrations add CLI args for runtime/infra concerns (tunnels, publishing,
    logging) that don't change the model, so they declare those flags via
    ``BaseIntegration.hash_exclusions()``. Best-effort and lazily imported: if
    the loader isn't importable or populated yet, this contributes nothing.
    """
    try:
        from praxis.cli import integration_loader

        return integration_loader.get_hash_exclusions()
    except Exception:
        return []


def resolve_exclude_from_hash(exclude_from_hash=None):
    """Effective exclusion list: the static defaults (or a caller-supplied base)
    plus anything loaded integrations declared. Integration exclusions are
    always merged so an integration flag never silently changes a run's hash."""
    base = list(
        DEFAULT_EXCLUDE_FROM_HASH if exclude_from_hash is None else exclude_from_hash
    )
    for flag in _integration_hash_exclusions():
        if flag and flag not in base:
            base.append(flag)
    return base


def compute_args_hash(args_list, exclude_from_hash=None):
    """
    Compute a deterministic hash from a list of command-line arguments.

    Args:
        args_list: List of command-line arguments (without script name)
        exclude_from_hash: List of argument names to exclude from hashing.
            Integration-declared exclusions are merged in regardless.

    Returns:
        str: SHA256 hash of the normalized arguments
    """
    exclude_from_hash = resolve_exclude_from_hash(exclude_from_hash)

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
