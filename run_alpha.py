#!/usr/bin/env python3
"""
Extended version of run.py with alpha configuration defaults.

This script modifies the CLI defaults before running the main training logic,
allowing for a base "alpha" configuration while preserving all CLI functionality.
"""

import os
import sys

# Ensure the script can find the local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Set up alpha defaults and run the main training script."""
    
    # Apply alpha configuration defaults to the CLI parser
    # We need to do this BEFORE any imports that use the CLI module
    
    # Import the CLI parser before it gets used by run.py
    import cli
    
    # Alpha configuration defaults that override the standard defaults
    alpha_defaults = {
        'seed': 42,
        'device': 'cuda:0', 
        'batch_size': 16,
        'depth': 3,
        'num_experts': 3,
        'vocab_size': 4096,
        'attention_type': 'standard',
        'strategy': 'naive',
        'tie_weights': True,
        'schedule_free': True,
    }
    
    # Update the parser's default values
    for action in cli.parser._actions:
        if hasattr(action, 'dest') and action.dest in alpha_defaults:
            # Convert underscores to hyphens for argument names (argparse converts them back)
            action.default = alpha_defaults[action.dest]
    
    # Re-parse arguments with updated defaults
    # This ensures the alpha defaults are used unless overridden by CLI args
    cli.args = cli.parser.parse_args()
    
    # Reconstruct the equivalent run.py command line for proper hash computation
    # This ensures that run_alpha.py produces the same hash as run.py with identical configs
    
    # We need to determine which arguments are actually different from run.py's original defaults
    # to reconstruct the minimal equivalent command
    
    # Get the original parser defaults before we modified them
    import argparse
    original_parser = argparse.ArgumentParser()
    
    # Get original defaults by finding them in the cli module
    # We'll reconstruct just the arguments that differ from original defaults + user args
    original_args = sys.argv[1:]  # Get the actual user-provided arguments
    
    # Build the equivalent run.py command by including:
    # 1. All alpha defaults that differ from original run.py defaults
    # 2. All user-provided arguments
    alpha_explicit_args = []
    
    # Add the alpha defaults as explicit arguments (these would need to be explicit in run.py)
    alpha_defaults = {
        'seed': 42,
        'device': 'cuda:0', 
        'batch_size': 16,
        'depth': 3,
        'num_experts': 3,
        'vocab_size': 4096,
        'attention_type': 'standard',
        'strategy': 'naive',
        'tie_weights': True,
        'schedule_free': True,
    }
    
    # Convert alpha defaults to CLI args
    for arg_name, value in alpha_defaults.items():
        cli_arg = '--' + arg_name.replace('_', '-')
        if isinstance(value, bool):
            if value:  # Only add flag if it's True
                alpha_explicit_args.append(cli_arg)
        else:
            alpha_explicit_args.extend([cli_arg, str(value)])
    
    # Combine alpha defaults with user args, removing duplicates (user args take precedence)
    user_arg_names = set()
    i = 0
    while i < len(original_args):
        if original_args[i].startswith('--'):
            user_arg_names.add(original_args[i])
            # Skip the value if this argument takes one
            if i + 1 < len(original_args) and not original_args[i + 1].startswith('-'):
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    # Filter out alpha args that user has overridden
    filtered_alpha_args = []
    i = 0
    while i < len(alpha_explicit_args):
        if alpha_explicit_args[i] not in user_arg_names:
            filtered_alpha_args.append(alpha_explicit_args[i])
            # Add value if present
            if i + 1 < len(alpha_explicit_args) and not alpha_explicit_args[i + 1].startswith('-'):
                filtered_alpha_args.append(alpha_explicit_args[i + 1])
                i += 2
            else:
                i += 1
        else:
            # Skip this alpha arg since user provided it
            if i + 1 < len(alpha_explicit_args) and not alpha_explicit_args[i + 1].startswith('-'):
                i += 2
            else:
                i += 1
    
    # Combine filtered alpha args with user args
    reconstructed_args = filtered_alpha_args + original_args
    
    # Save the original command for logging
    original_command = sys.argv[:]
    
    # Replace sys.argv to compute the correct hash
    sys.argv = ['run.py'] + reconstructed_args
    
    # We need to compute the hash from expanded args but log the original command
    # and ensure we only log once (not twice)
    
    # Import cli to set up custom logging
    import cli
    
    # Compute the hash from the expanded arguments
    def compute_hash_from_expanded_args():
        import json
        import hashlib
        
        exclude_from_hash = ["--reset", "--debug"]
        args = reconstructed_args
        
        # Create a normalized representation of arguments for hashing
        arg_dict = {}
        i = 0
        while i < len(args):
            if args[i].startswith("-"):
                arg_name = args[i]
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    if arg_name not in exclude_from_hash:
                        arg_dict[arg_name] = args[i + 1]
                    i += 2
                else:
                    if arg_name not in exclude_from_hash:
                        arg_dict[arg_name] = True
                    i += 1
            else:
                pos_arg_name = f"_pos_{i}"
                if pos_arg_name not in exclude_from_hash:
                    arg_dict[pos_arg_name] = args[i]
                i += 1
        
        sorted_args = dict(sorted(arg_dict.items()))
        args_json = json.dumps(sorted_args, sort_keys=True)
        hash_object = hashlib.sha256(args_json.encode())
        return hash_object.hexdigest()
    
    computed_hash = compute_hash_from_expanded_args()
    truncated_hash = computed_hash[:9]
    
    # Create a custom log_command that uses our pre-computed hash and original command
    original_log_command = cli.log_command
    has_logged = False
    
    def custom_log_command(exclude_from_hash=["--reset", "--debug"]):
        nonlocal has_logged
        if has_logged:
            # Return cached values to prevent double logging
            return "", computed_hash, truncated_hash
        
        has_logged = True
        
        # Log the original command with our computed hash
        import os
        from datetime import datetime
        
        script_name = os.path.basename(original_command[0])
        args = original_command[1:]
        full_command = f"python {script_name} {' '.join(args)}"
        
        # Format log entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        
        # Save the truncated hash to MODEL_HASH.txt
        hash_file_dir = os.path.join("data", "praxis")
        hash_file_path = os.path.join(hash_file_dir, "MODEL_HASH.txt")
        os.makedirs(hash_file_dir, exist_ok=True)
        with open(hash_file_path, "w") as f:
            f.write(truncated_hash)
        
        return full_command, computed_hash, truncated_hash
    
    # Replace the log_command function
    cli.log_command = custom_log_command
    
    # Restore original command for parsing
    sys.argv = original_command
    
    # Parse args with the alpha defaults applied (they're already set in the parser)
    cli.args = cli.parser.parse_args()
    
    # Now import and run the main script logic
    try:
        import run
        # The run module executes immediately upon import, so we're done
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\n🛑 Training interrupted by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        # Re-raise any other exceptions
        raise


if __name__ == "__main__":
    main()