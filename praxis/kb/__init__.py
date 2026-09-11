"""Praxis knowledge base: a generic, type-driven search bus.

Sources (the ``kb_sources`` registry) emit normalized ``KBItem``s; ``KBIndex``
indexes them once and answers ranked queries per keystroke. Backs the
Gymnasium "Read" search and any future relevance-ranking task.
"""

from praxis.kb.index import DEFAULT_DB_PATH, KBIndex
from praxis.kb.item import KBHit, KBItem
from praxis.kb.sources import KBSource

__all__ = [
    "KBItem",
    "KBHit",
    "KBSource",
    "KBIndex",
    "DEFAULT_DB_PATH",
]
