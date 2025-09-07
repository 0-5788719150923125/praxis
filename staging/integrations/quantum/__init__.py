"""Quantum code dataset module for Praxis."""

# Import the Integration class directly
from .integration import Integration

# Also keep backward compatibility exports from main.py
from .main import add_cli_args, initialize, provide_dataset