"""
Utility functions for Praxis.

This module re-exports all utilities for backward compatibility.
"""

# Re-export everything from submodules for backward compatibility
from praxis.utils.arrays import (
    coerce_to_list,
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
    configure_cuda_allocator,
    configure_multiprocessing,
    find_latest_checkpoint,
    graceful_shutdown,
    initialize_lazy_modules,
    is_shutting_down,
    perform_reset,
    register_child_process,
    register_cleanup_function,
    resolve_resume_checkpoint,
    show_launch_animation,
    shutdown_manager,
    sigint_handler,
    update_license_timestamp,
)
from praxis.utils.tensors import create_block_ids, norm_scaling

# Make all exports available at module level
__all__ = [
    # Arrays
    "coerce_to_list",
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
    "shutdown_manager",
    "register_cleanup_function",
    "register_child_process",
    "is_shutting_down",
    "check_for_updates",
    "configure_cuda_allocator",
    "configure_multiprocessing",
    "find_latest_checkpoint",
    "resolve_resume_checkpoint",
    "graceful_shutdown",
    "update_license_timestamp",
    "initialize_lazy_modules",
    "perform_reset",
    "show_launch_animation",
]
