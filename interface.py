"""Backward compatibility wrapper for interface module.

This file maintains backward compatibility with the old interface.py location.
The actual implementation has been moved to praxis.interface module.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'interface' is deprecated. "
    "Please import from 'praxis.interface' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location
from praxis.interface import (
    TerminalDashboard,
    get_active_dashboard,
    register_socketio
)

__all__ = ['TerminalDashboard', 'get_active_dashboard', 'register_socketio']