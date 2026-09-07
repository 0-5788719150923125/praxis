"""Hash computation for CLI arguments."""

import hashlib
import json

# Flags declared with ``exclude_hash=True`` at their add_argument() call site
# (see PraxisArgumentParser in parser.py). Insertion-ordered, used as a set.
#
# This replaced a hand-written list of flag strings that had to be kept in sync
# with the arguments by hand - the two drifted, and a missed entry silently
# forks a run's identity (and its checkpoint directory) the first time someone
# passes the flag.
_HASH_EXCLUSIONS = {}

# Whether we have gone looking for argument definitions (see _ensure_declared).
_declared = False


def register_hash_exclusion(*flags):
    """Record flags that must not contribute to the run hash.

    Called by the parser as each argument is defined; not meant to be called
    directly - declare ``exclude_hash=True`` on the argument instead.
    """
    for flag in flags:
        if flag:
            _HASH_EXCLUSIONS[flag] = None


def _ensure_declared():
    """Make sure the argument definitions have actually been executed.

    On the normal path initialize_cli() built the parser long before anything
    asks for a hash, so the registry is already full. The exception is callers
    that hash argv without owning it - tests and tools importing
    compute_args_hash directly. Registration is a side effect of DEFINING the
    arguments, so define them against a throwaway parser: nothing is parsed,
    nothing is kept, and integrations (which only register when loaded)
    contribute nothing here, exactly as before.
    """
    global _declared
    if _declared or _HASH_EXCLUSIONS:
        return
    _declared = True
    try:
        from praxis.cli.core.parser import create_base_parser
        from praxis.cli.groups import add_all_argument_groups

        add_all_argument_groups(create_base_parser())
    except Exception:
        pass


def declared_hash_exclusions():
    """Every flag declared with ``exclude_hash=True``, in registration order."""
    _ensure_declared()
    return list(_HASH_EXCLUSIONS)


def resolve_exclude_from_hash(exclude_from_hash=None):
    """Effective exclusion list: everything declared at an add_argument() site,
    plus any extras a caller passes. The declared set is always merged in - a
    caller supplying its own list must not be able to quietly pull a runtime
    flag back into a run's identity."""
    merged = dict.fromkeys(declared_hash_exclusions())
    for flag in exclude_from_hash or []:
        merged.setdefault(flag, None)
    return list(merged)


def compute_args_hash(args_list, exclude_from_hash=None):
    """
    Compute a deterministic hash from a list of command-line arguments.

    Args:
        args_list: List of command-line arguments (without script name)
        exclude_from_hash: Extra argument names to exclude from hashing.
            Flags declared with ``exclude_hash=True`` are merged in regardless.

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
