"""CALM encoder package: token-chunk VAE + energy head.

Paper: Continuous Autoregressive Language Models (arXiv 2510.27688).
See ``PLAN.md`` for the high-level design and how each piece maps onto
the existing Praxis encoder interface.
"""

from .encoder import CALMEncoder
from .vae import CALMVAE

__all__ = ["CALMEncoder", "CALMVAE"]
