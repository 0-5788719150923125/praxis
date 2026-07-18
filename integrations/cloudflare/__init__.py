"""Cloudflare Pages integration for Praxis.

Publishes a static snapshot of the live dashboard - the built frontend plus a
dump of every read-only endpoint - to Cloudflare Pages, so the showcase stays
online for free after the training server shuts down.
"""

from .main import Integration

__all__ = ["Integration"]
