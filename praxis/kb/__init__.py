"""Praxis knowledge base: a generic, type-driven search bus.

Sources (``KB_SOURCE_REGISTRY``) emit normalized ``KBItem``s; ``KBIndex``
indexes them once and answers ranked queries per keystroke. Backs the
Gymnasium "Read" search and any future relevance-ranking task.
"""

from praxis.kb.index import DEFAULT_DB_PATH, KBIndex
from praxis.kb.item import KBHit, KBItem
from praxis.kb.sources import KB_SOURCE_REGISTRY, KBSource

__all__ = [
    "KBItem",
    "KBHit",
    "KBSource",
    "KB_SOURCE_REGISTRY",
    "KBIndex",
    "DEFAULT_DB_PATH",
]
