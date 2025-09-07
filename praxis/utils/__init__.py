"""
Utility functions for Praxis.

This module re-exports all utilities for backward compatibility.
"""

# Re-export everything from submodules for backward compatibility
from praxis.utils.arrays import (
    generate_alternating_values,
    generate_decay_values,
    generate_u_shape_values,
)
from praxis.utils.memory import get_memory_info
from praxis.utils.naming import (
    PREFIXES,
    SUFFIXES,
    generate_deterministic_name,
    mask_git_url,
)
from praxis.utils.system import (
    check_for_updates,
    find_latest_checkpoint,
    initialize_lazy_modules,
    sigint_handler,
)
from praxis.utils.tensors import create_block_ids, norm_scaling

# Make all exports available at module level
__all__ = [
    # Arrays
    "generate_alternating_values",
    "generate_decay_values",
    "generate_u_shape_values",
    # Tensors
    "norm_scaling",
    "create_block_ids",
    # Naming
    "PREFIXES",
    "SUFFIXES",
    "generate_deterministic_name",
    "mask_git_url",
    # Memory
    "get_memory_info",
    # System
    "sigint_handler",
    "check_for_updates",
    "find_latest_checkpoint",
    "initialize_lazy_modules",
]
