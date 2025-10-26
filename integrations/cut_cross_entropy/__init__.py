"""Cut Cross-Entropy loss integration for Praxis.

This integration provides Apple's memory-efficient cut-cross-entropy loss function,
which avoids materializing full logit matrices and shifted tensor copies.
"""

# Import the Integration class from main.py
from .main import Integration

# Export the Integration class
__all__ = ["Integration"]
