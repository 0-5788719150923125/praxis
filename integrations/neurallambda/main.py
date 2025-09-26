"""NeuralLambda integration for Praxis."""

import argparse


# Auto-register the neural controller when module is imported
def _register_controller():
    """Register neural controller in the global registry."""
    try:
        from praxis.controllers import CONTROLLER_REGISTRY
        from .neural import NeuralController

        CONTROLLER_REGISTRY["neural"] = NeuralController
    except ImportError:
        # Fallback for when imported directly
        import sys
        import os

        integration_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, integration_dir)
        try:
            from praxis.controllers import CONTROLLER_REGISTRY
            from neural import NeuralController

            CONTROLLER_REGISTRY["neural"] = NeuralController
        except ImportError:
            # Silently fail if registry not available yet
            pass


# Register controller immediately on import
_register_controller()


class Integration:
    """NeuralLambda integration for Praxis."""

    def __init__(self):
        """Initialize the NeuralLambda integration."""
        self.name = "neurallambda"

    def add_arguments(self, parser: argparse.ArgumentParser):
        """No additional arguments needed."""
        return parser

    def lifecycle(self, stage: str, *args, **kwargs):
        """Handle integration lifecycle events."""
        # Controller is already registered during import
        return None
