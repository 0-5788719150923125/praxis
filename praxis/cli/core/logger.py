"""Command logging functionality."""

import os
import sys
from datetime import datetime

from .hasher import compute_args_hash, DEFAULT_EXCLUDE_FROM_HASH


def log_command(exclude_from_hash=None, custom_command=None, custom_hash=None):
    """
    Logs the current command line execution to history.log in the root directory.
    New commands are added to the top of the file.

    Args:
        exclude_from_hash: List of argument names to exclude from hashing
        custom_command: Optional custom command string to log
        custom_hash: Optional pre-computed hash to use

    Returns:
        tuple: (full_command, args_hash, truncated_hash)
    """
    # Use default exclude list if None provided
    if exclude_from_hash is None:
        exclude_from_hash = DEFAULT_EXCLUDE_FROM_HASH

    # Construct the command
    if custom_command:
        full_command = custom_command
    else:
        script_name = os.path.basename(sys.argv[0])
        args = sys.argv[1:]
        full_command = f"python {script_name} {' '.join(args)}"

    # Compute hash if not provided
    if custom_hash:
        args_hash = custom_hash
    else:
        args = sys.argv[1:] if not custom_command else []
        args_hash = compute_args_hash(args, exclude_from_hash)

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