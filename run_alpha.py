#!/usr/bin/env python3
"""
Extended version of run.py with alpha configuration defaults.

This script applies alpha defaults and then runs the main training logic,
maintaining full CLI compatibility while ensuring consistent model tracking.
"""

import os
import sys

# Ensure the script can find the local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Set up alpha defaults and run the main training script."""
    
    # Import CLI utilities
    import cli
    
    # Define alpha configuration defaults
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
    
    # Apply alpha defaults and get properly configured args
    # This handles all the hash computation and logging automatically
    cli.args, effective_hash = cli.apply_defaults_and_parse(alpha_defaults)
    
    # Now import and run the main script logic
    try:
        import run
        # The run module executes immediately upon import
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        raise


if __name__ == "__main__":
    main()