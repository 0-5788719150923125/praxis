"""Hivemind integration module for Praxis."""

# Import the Integration class from main.py
from .main import Integration, get_hivemind_errors, get_hivemind_manager

# Export the Integration class and utility functions
__all__ = ["Integration", "get_hivemind_manager", "get_hivemind_errors"]
